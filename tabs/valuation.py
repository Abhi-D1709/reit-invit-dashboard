# tabs/valuation.py
import re
import time
import html
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas._libs.tslibs.nattype import NaTType  # <-- for typing NaT correctly
import requests
from bs4 import BeautifulSoup  # type: ignore
import streamlit as st

# --------------------------------------------------------------------
# Config: read the Google Sheet URL from utils.common (no hardcoding)
# --------------------------------------------------------------------
try:
    from utils.common import VALUATION_REIT_SHEET_URL  # type: ignore
    SHEET_URL = str(VALUATION_REIT_SHEET_URL).strip()
except Exception:
    SHEET_URL = ""

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def _doc_id_from_share_url(url: str) -> Optional[str]:
    m = re.search(r"/d/([^/]+)/", url or "")
    return m.group(1) if m else None

def _csv_export_url(share_url: str, sheet_name: str) -> str:
    """
    Build a CSV export URL for a given sheet name (title).
    """
    doc_id = _doc_id_from_share_url(share_url)
    if not doc_id:
        return ""
    return (
        f"https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq"
        f"?tqx=out:csv&sheet={requests.utils.quote(sheet_name)}"
    )

def _parse_date(value: Any) -> pd.Timestamp | NaTType:
    """
    Parse dates like '14/09/2020', '-', 'NA', etc. Assume day-first.
    """
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.to_datetime(value)
    if value is None:
        return pd.NaT
    s = str(value).strip()
    if not s or s.upper() == "NA" or s == "-":
        return pd.NaT
    s = re.sub(r"\s+", " ", html.unescape(s))
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def _years_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    return float((d2 - d1).days) / 365.25

def _normalize_name(x: str) -> str:
    s = (x or "").lower()
    s = re.sub(r"\b(mr|mrs|ms|dr|shri|smt|kum)\.?\s+", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --------------------------------------------------------------------
# Data loaders (cache heavily)
# --------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60)  # 6 hours
def load_valuers_sheet(sheet_url: str) -> pd.DataFrame:
    """
    Reads Sheet1:
      - Name of REIT
      - Finanical Year / Financial Year (support both)
      - Name of Valuer
      - Date of Appointmnet (original header spelling)
      - Date of Resignation
      - IBBI Registration No.
    """
    csv_url = _csv_export_url(sheet_url, "Sheet1")
    if not csv_url:
        return pd.DataFrame()

    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]

    # Normalize the FY header if needed
    if "Financial Year" in df.columns and "Finanical Year" not in df.columns:
        df.rename(columns={"Financial Year": "Finanical Year"}, inplace=True)

    expected = [
        "Name of REIT",
        "Finanical Year",
        "Name of Valuer",
        "Date of Appointmnet",
        "Date of Resignation",
        "IBBI Registration No.",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA

    df["Date of Appointmnet"] = df["Date of Appointmnet"].apply(_parse_date)
    df["Date of Resignation"] = df["Date of Resignation"].apply(_parse_date)

    return df

def _extract_table_rows(tbl: BeautifulSoup) -> List[Dict[str, str]]:
    headers: List[str] = []
    rows: List[Dict[str, str]] = []

    thead = tbl.find("thead")
    if thead:
        ths = thead.find_all("th")
        headers = [th.get_text(strip=True) for th in ths]

    tbody = tbl.find("tbody")
    if not tbody:
        return rows

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        values = [td.get_text(strip=True) for td in tds]
        if headers and len(values) == len(headers):
            row = {headers[i]: values[i] for i in range(len(headers))}
        else:
            row = {f"col_{i}": values[i] for i in range(len(values))}
        rows.append(row)
    return rows

def _find_max_page(soup: BeautifulSoup) -> Optional[int]:
    """
    Try to find the last page number from pagination links (best effort).
    """
    max_page: Optional[int] = None
    for a in soup.find_all("a", href=True):
        m = re.search(r"[?&]page=(\d+)\b", a["href"])
        if m:
            p = int(m.group(1))
            max_page = p if (max_page is None or p > max_page) else max_page
    return max_page

def _scrape_table_pages(base_url: str, kind: str, max_pages: Optional[int], concurrency: int) -> List[Dict[str, str]]:
    """
    Crawl pages concurrently. If max_pages is None, try to detect last page
    from page 1; otherwise limit to max_pages.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (REIT-INVIT-Dashboard)"})
    rows_all: List[Dict[str, str]] = []

    # Fetch page 1 to detect last page (if needed)
    r1 = session.get(base_url.format(page=1), timeout=30)
    if r1.status_code != 200:
        return rows_all
    soup1 = BeautifulSoup(r1.text, "lxml")
    first_rows: List[Dict[str, str]] = []
    for tbl in soup1.find_all("table"):
        first_rows.extend(_extract_table_rows(tbl))

    # Determine how many pages to fetch
    total_pages = max_pages
    if total_pages is None:
        last = _find_max_page(soup1)
        total_pages = last if last and last > 1 else 1

    # Collect pages to fetch
    pages = list(range(1, total_pages + 1))
    results: Dict[int, List[Dict[str, str]]] = {}
    results[1] = first_rows

    remaining = [p for p in pages if p != 1]
    if remaining:
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, 12))) as ex:
            fut_to_p = {
                ex.submit(session.get, base_url.format(page=p), 30): p
                for p in remaining
            }
            prog = st.progress(0.0, text=f"Fetching {kind} registry …")
            done_n = 0
            for fut in as_completed(fut_to_p):
                p = fut_to_p[fut]
                try:
                    resp = fut.result()
                    lst: List[Dict[str, str]] = []
                    if resp.status_code == 200:
                        s = BeautifulSoup(resp.text, "lxml")
                        for tbl in s.find_all("table"):
                            lst.extend(_extract_table_rows(tbl))
                    results[p] = lst
                except Exception:
                    results[p] = []
                done_n += 1
                prog.progress(done_n / len(remaining))
            prog.empty()

    # Flatten preserving page order
    for p in pages:
        rows_all.extend(results.get(p, []))

    # Normalize columns to the two we need
    normed: List[Dict[str, str]] = []
    for r in rows_all:
        keys = {k.lower(): k for k in r.keys()}
        reg_key = (
            keys.get("registration no.")
            or keys.get("registration no")
            or keys.get("reg. no.")
            or keys.get("reg no.")
        )
        name_key = (
            keys.get("name of rv")
            or keys.get("name of rve")
            or keys.get("name")
        )
        if reg_key and name_key:
            normed.append(
                {
                    "source": kind,
                    "Registration No.": r.get(reg_key, "").strip(),
                    "Name": r.get(name_key, "").strip(),
                }
            )
    return normed

@st.cache_data(ttl=7 * 24 * 60 * 60)  # 7 days
def scrape_ibbi_registry_cached(mode: str, max_pages_ind: int, max_pages_ent: int, concurrency: int) -> pd.DataFrame:
    """
    Build the registry once and cache for 7 days.
    mode: 'fast' (limit pages) or 'full' (auto-detect all pages).
    """
    ind_url = "https://ibbi.gov.in/service-provider/rvs?page={page}"
    ent_url = "https://ibbi.gov.in/service-provider/rvo-entities?page={page}"

    if mode == "fast":
        max_ind: Optional[int] = max_pages_ind
        max_ent: Optional[int] = max_pages_ent
    else:
        max_ind = None
        max_ent = None

    ind_rows = _scrape_table_pages(ind_url, "individual", max_ind, concurrency)
    ent_rows = _scrape_table_pages(ent_url, "entity", max_ent, concurrency)

    df = pd.DataFrame(ind_rows + ent_rows)
    if df.empty:
        return df

    df["norm_name"] = df["Name"].map(_normalize_name)
    df["reg_norm"] = df["Registration No."].str.replace(r"\s+", "", regex=True).str.upper()
    df.drop_duplicates(subset=["reg_norm", "norm_name"], inplace=True)
    return df

# --------------------------------------------------------------------
# Business rules
# --------------------------------------------------------------------
def check_tenure_leq_4yrs(appoint: pd.Timestamp, resign: pd.Timestamp | NaTType) -> tuple[bool, Optional[float]]:
    if isinstance(appoint, pd.Timestamp) and not pd.isna(appoint):
        end_dt = resign if isinstance(resign, pd.Timestamp) and not pd.isna(resign) else pd.Timestamp(date.today())
        yrs = _years_between(appoint, end_dt)
        return yrs <= 4.0, yrs
    return False, None

def check_ibbi_registered(name: str, reg_no: str, registry: pd.DataFrame) -> bool:
    """
    Prefer exact Registration No. match; fall back to normalized name.
    """
    if registry is None or registry.empty:
        return False
    reg_norm = (reg_no or "").replace(" ", "").upper()
    if reg_norm:
        if (registry["reg_norm"] == reg_norm).any():
            return True
    nm = _normalize_name(name or "")
    if not nm:
        return False
    return (registry["norm_name"] == nm).any()

# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
def render_valuation() -> None:
    st.header("Valuation")

    if not SHEET_URL:
        st.warning("Add VALUATION_REIT_SHEET_URL to utils/common.py (public view link to your workbook).")
        return

    with st.sidebar:
        st.subheader("Valuation — Settings")
        mode = st.radio(
            "IBBI registry mode",
            ["Fast (cached, limited pages)", "Full (cached, all pages)"],
            index=0,
            help="Fast scans only the first N pages and caches for 7 days. Full scans all pages (slower) and caches for 7 days.",
        )
        fast = mode.startswith("Fast")
        max_pages_ind = st.number_input("Individuals: pages to scan (fast mode)", 1, 400, 60, help="Applies only in fast mode.")
        max_pages_ent = st.number_input("Entities: pages to scan (fast mode)", 1, 50, 10, help="Applies only in fast mode.")
        concurrency = st.slider("Network concurrency", 1, 12, 6, help="Higher is faster but may stress the site.")
        refresh = st.button("Refresh IBBI registry cache")

    if refresh:
        scrape_ibbi_registry_cached.clear()  # clear cache for the function
        st.success("Registry cache cleared. It will be rebuilt on the next run.")

    # Load valuation rows
    val_df = load_valuers_sheet(SHEET_URL)
    if val_df.empty:
        st.info("No valuation records found in Sheet1.")
        return

    # Filters
    reits = sorted([x for x in val_df["Name of REIT"].dropna().unique().tolist() if str(x).strip()])
    c1, c2 = st.columns([1, 1])
    with c1:
        selected_reit = st.selectbox("Select REIT", reits, index=0 if reits else None)
    with c2:
        fy_opts = (
            val_df.loc[val_df["Name of REIT"] == selected_reit, "Finanical Year"]
            .dropna().astype(str).unique().tolist()
        )
        fy_opts = sorted(fy_opts)
        selected_fy = st.selectbox("Financial Year", fy_opts, index=0 if fy_opts else None)

    filtered = val_df[
        (val_df["Name of REIT"] == selected_reit)
        & (val_df["Finanical Year"].astype(str) == str(selected_fy))
    ].copy()

    if filtered.empty:
        st.warning("No rows for the selected REIT and year.")
        return

    # Pull IBBI registry once (cached 7 days)
    with st.spinner("Loading IBBI registry …"):
        registry = scrape_ibbi_registry_cached(
            mode="fast" if fast else "full",
            max_pages_ind=int(max_pages_ind),
            max_pages_ent=int(max_pages_ent),
            concurrency=int(concurrency),
        )

    # Evaluate rows
    out_rows: List[Dict[str, Any]] = []
    for _, r in filtered.iterrows():
        name = str(r.get("Name of Valuer", "")).strip()
        regno = str(r.get("IBBI Registration No.", "")).strip()
        appoint = r.get("Date of Appointmnet")
        resign = r.get("Date of Resignation")

        ok_tenure, yrs = check_tenure_leq_4yrs(appoint, resign)
        ok_reg = check_ibbi_registered(name, regno, registry)

        out_rows.append(
            {
                "Name of REIT": r.get("Name of REIT"),
                "Financial Year": r.get("Finanical Year"),
                "Valuer": name,
                "IBBI Reg No.": regno,
                "Appointment": appoint.date().isoformat() if isinstance(appoint, pd.Timestamp) and not pd.isna(appoint) else "",
                "Resignation": resign.date().isoformat() if isinstance(resign, pd.Timestamp) and not pd.isna(resign) else "",
                "Tenure ≤ 4 years": "✅" if ok_tenure else "❌",
                "Tenure (yrs)": f"{yrs:.2f}" if yrs is not None else "",
                "Registered with IBBI": "✅" if ok_reg else "❌",
            }
        )

    res_df = pd.DataFrame(out_rows)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Rows analysed", len(res_df))
    with m2:
        st.metric("Tenure OK (≤ 4 yrs)", int((res_df["Tenure ≤ 4 years"] == "✅").sum()))
    with m3:
        st.metric("Registered with IBBI", int((res_df["Registered with IBBI"] == "✅").sum()))

    st.dataframe(res_df, use_container_width=True, hide_index=True)

# Backward-compatible entry point for your page wrapper
def render():
    render_valuation()
