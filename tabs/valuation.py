# tabs/valuation.py
import re
import html
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas._libs.tslibs.nattype import NaTType
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup  # type: ignore
import streamlit as st

# ------------------------------------------------------------
# Config: read public sheet URL from utils.common (no hardcode)
# ------------------------------------------------------------
try:
    from utils.common import VALUATION_REIT_SHEET_URL  # type: ignore
    SHEET_URL = str(VALUATION_REIT_SHEET_URL).strip()
except Exception:
    SHEET_URL = ""

# ------------------------------------------------------------
# “Full blast” assumptions (change if IBBI pagination changes)
# ------------------------------------------------------------
INDIVIDUAL_PAGES = 303   # https://ibbi.gov.in/service-provider/rvs?page=1..N
ENTITY_PAGES     = 7     # https://ibbi.gov.in/service-provider/rvo-entities?page=1..N

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _doc_id_from_share_url(url: str) -> Optional[str]:
    m = re.search(r"/d/([^/]+)/", url or "")
    return m.group(1) if m else None

def _csv_export_url(share_url: str, sheet_name: str) -> str:
    doc_id = _doc_id_from_share_url(share_url)
    if not doc_id:
        return ""
    return (
        f"https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq"
        f"?tqx=out:csv&sheet={requests.utils.quote(sheet_name)}"
    )

def _parse_date(value: Any) -> pd.Timestamp | NaTType:
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

# ------------------------------------------------------------
# Load Sheet1 (Valuers)
# ------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60)
def load_valuers_sheet(sheet_url: str) -> pd.DataFrame:
    csv_url = _csv_export_url(sheet_url, "Sheet1")
    if not csv_url:
        return pd.DataFrame()

    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]

    # Align header spelling
    if "Financial Year" in df.columns and "Finanical Year" not in df.columns:
        df.rename(columns={"Financial Year": "Finanical Year"}, inplace=True)

    expected = [
        "Name of REIT",
        "Finanical Year",
        "Name of Valuer",
        "Date of Appointmnet",   # keep original header spelling
        "Date of Resignation",
        "IBBI Registration No.",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA

    df["Date of Appointmnet"] = df["Date of Appointmnet"].apply(_parse_date)
    df["Date of Resignation"] = df["Date of Resignation"].apply(_parse_date)

    return df

# ------------------------------------------------------------
# IBBI registry scraping — 310-at-once with threads
# ------------------------------------------------------------
def _http_session(connections: int) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (REIT-INVIT-Dashboard)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=connections, pool_maxsize=connections, max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def _extract_table_rows(tbl: BeautifulSoup) -> List[Dict[str, str]]:
    headers: List[str] = []
    rows: List[Dict[str, str]] = []
    thead = tbl.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
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

def _parse_registry_html(html_text: str, kind: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    rows_all: List[Dict[str, str]] = []
    for tbl in soup.find_all("table"):
        rows_all.extend(_extract_table_rows(tbl))
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

def _fetch_page(session: requests.Session, url: str, page: int) -> Tuple[int, Optional[str]]:
    try:
        r = session.get(url.format(page=page), timeout=30)
        if r.status_code == 200 and "<html" in r.text.lower():
            return page, r.text
        return page, None
    except Exception:
        return page, None

@st.cache_data(ttl=7 * 24 * 60 * 60)
def build_ibbi_registry_fullblast(ind_pages: int, ent_pages: int) -> pd.DataFrame:
    """
    Fire ALL pages at once (ind_pages + ent_pages workers).
    Cached for 7 days.
    """
    base_ind = "https://ibbi.gov.in/service-provider/rvs?page={page}"
    base_ent = "https://ibbi.gov.in/service-provider/rvo-entities?page={page}"

    total = ind_pages + ent_pages
    session = _http_session(connections=max(32, total))  # large pool

    # Prepare page lists
    pages_ind = list(range(1, ind_pages + 1))
    pages_ent = list(range(1, ent_pages + 1))

    # Blast with a thread per page (bounded by total)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows_normed: List[Dict[str, str]] = []
    prog = st.progress(0.0, text="Fetching IBBI registry (full blast)…")
    done = 0

    with ThreadPoolExecutor(max_workers=total) as ex:
        futs = []
        for p in pages_ind:
            futs.append(ex.submit(_fetch_page, session, base_ind, p))
        for p in pages_ent:
            futs.append(ex.submit(_fetch_page, session, base_ent, p))

        for fut in as_completed(futs):
            page, html_text = fut.result()
            if html_text:
                # Determine kind by URL template matched
                kind = "individual" if page in pages_ind else "entity"
                rows_normed.extend(_parse_registry_html(html_text, kind))
            done += 1
            prog.progress(done / total)

    prog.empty()

    df = pd.DataFrame(rows_normed)
    if df.empty:
        return df
    df["norm_name"] = df["Name"].map(_normalize_name)
    df["reg_norm"] = df["Registration No."].str.replace(r"\s+", "", regex=True).str.upper()
    df.drop_duplicates(subset=["reg_norm", "norm_name"], inplace=True)
    return df

# ------------------------------------------------------------
# Business rules
# ------------------------------------------------------------
def check_tenure_leq_4yrs(appoint: pd.Timestamp, resign: pd.Timestamp | NaTType) -> tuple[bool, Optional[float]]:
    if isinstance(appoint, pd.Timestamp) and not pd.isna(appoint):
        end_dt = resign if isinstance(resign, pd.Timestamp) and not pd.isna(resign) else pd.Timestamp(date.today())
        yrs = _years_between(appoint, end_dt)
        return yrs <= 4.0, yrs
    return False, None

def check_ibbi_registered(name: str, reg_no: str, registry: pd.DataFrame) -> bool:
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

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
def render() -> None:
    st.header("Valuation")

    if not SHEET_URL:
        st.warning("Add VALUATION_REIT_SHEET_URL to utils/common.py (public view link).")
        return

    with st.sidebar:
        st.subheader("IBBI Registry (Full Blast)")
        st.caption("Runs all pages concurrently and caches the merged registry for 7 days.")
        ind_pages = st.number_input("Individual RV pages", 1, 1000, INDIVIDUAL_PAGES, step=1)
        ent_pages = st.number_input("Entity RV pages", 1, 100, ENTITY_PAGES, step=1)
        if st.button("Refresh registry now"):
            build_ibbi_registry_fullblast.clear()
            st.success("Cleared cache. It will rebuild below.")

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

    # Build the IBBI registry in one shot (cached)
    with st.spinner("Building IBBI registry (one-time, cached 7 days)…"):
        registry = build_ibbi_registry_fullblast(int(ind_pages), int(ent_pages))

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
