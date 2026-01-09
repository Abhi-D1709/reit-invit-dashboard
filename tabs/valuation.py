# tabs/valuation.py
from __future__ import annotations

import re
import html
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup  # type: ignore
import streamlit as st

# Supabase: import create_client at runtime; import Client only for type-checking.
try:
    from supabase import create_client  # type: ignore
except Exception:  # library not installed yet
    create_client = None

if TYPE_CHECKING:
    from supabase import Client  # type: ignore
else:
    Client = Any  # type: ignore

# ------------------------------------------------------------
# Config: read public sheet URL from utils.common (no hardcode)
# ------------------------------------------------------------
try:
    from utils.common import VALUATION_REIT_SHEET_URL  # type: ignore
    DEFAULT_SHEET_URL = str(VALUATION_REIT_SHEET_URL).strip()
except Exception:
    DEFAULT_SHEET_URL = ""

# Base listing endpoints
INDIVIDUAL_BASE = "https://ibbi.gov.in/service-provider/rvs?page={page}"
ENTITY_BASE     = "https://ibbi.gov.in/service-provider/rvo-entities?page={page}"

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

def _parse_date(value: Any) -> Optional[pd.Timestamp]:
    """Return pd.Timestamp or None (no NaT in signatures to keep Pylance happy)."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.to_datetime(value)
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() == "NA" or s == "-":
        return None
    s = re.sub(r"\s+", " ", html.unescape(s))
    ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return None if pd.isna(ts) else ts

def _years_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    return float((d2 - d1).days) / 365.25

def _normalize_name(x: str) -> str:
    s = (x or "").lower()
    s = re.sub(r"\b(mr|mrs|ms|dr|shri|smt|kum)\.?\s+", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _http_session(pool: int = 64) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (REIT-INVIT-Dashboard Valuation)",
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
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
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

    # Align header spelling from the Google Sheet (“Finanical Year”, “Appointmnet”)
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

# ------------------------------------------------------------
# IBBI registry scraping (auto-discover page counts)
# ------------------------------------------------------------
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
        name_key = keys.get("name of rv") or keys.get("name of rve") or keys.get("name")
        if reg_key and name_key:
            normed.append(
                {
                    "source": kind,
                    "Registration No.": r.get(reg_key, "").strip(),
                    "Name": r.get(name_key, "").strip(),
                }
            )
    return normed

def _fetch_page(session: requests.Session, url_tpl: str, page: int) -> Tuple[int, Optional[str]]:
    try:
        r = session.get(url_tpl.format(page=page), timeout=25)
        if r.status_code == 200 and "<html" in r.text.lower():
            return page, r.text
        return page, None
    except Exception:
        return page, None

def _discover_last_valid_page(
    session: requests.Session,
    url_tpl: str,
    kind_for_parse: str,
    start: int = 1,
    window: int = 30,
    max_page: int = 2000,
) -> int:
    """Walk in windows until a window has no tables. Return last valid page, 0 if none."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    last_valid = 0
    p = start
    while p <= max_page:
        batch = list(range(p, min(p + window - 1, max_page) + 1))
        with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as ex:
            futs = [ex.submit(_fetch_page, session, url_tpl, i) for i in batch]
            any_valid = False
            for fut in as_completed(futs):
                page, html_text = fut.result()
                if not html_text:
                    continue
                rows = _parse_registry_html(html_text, kind_for_parse)
                if rows:
                    any_valid = True
                    if page > last_valid:
                        last_valid = page
        if not any_valid:
            break
        p = batch[-1] + 1
    return last_valid

def _scrape_ibbi_registry_full() -> pd.DataFrame:
    """Auto-discover + fetch all pages concurrently, returns normalized df."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    session = _http_session(pool=64)
    ind_last = _discover_last_valid_page(session, INDIVIDUAL_BASE, "individual")
    ent_last = _discover_last_valid_page(session, ENTITY_BASE, "entity")

    total_pages = ind_last + ent_last
    rows_normed: List[Dict[str, str]] = []

    if total_pages == 0:
        return pd.DataFrame(columns=["source", "Registration No.", "Name", "norm_name", "reg_norm"])

    prog = st.progress(0.0, text=f"Fetching IBBI registry (pages: {ind_last} + {ent_last})…")
    done = 0

    with ThreadPoolExecutor(max_workers=min(total_pages, 64)) as ex:
        futs = []
        for p in range(1, ind_last + 1):
            futs.append(ex.submit(_fetch_page, session, INDIVIDUAL_BASE, p))
        for p in range(1, ent_last + 1):
            futs.append(ex.submit(_fetch_page, session, ENTITY_BASE, p))

        for fut in as_completed(futs):
            page, html_text = fut.result()
            if html_text:
                kind = "individual" if page <= ind_last else "entity"
                rows_normed.extend(_parse_registry_html(html_text, kind))
            done += 1
            prog.progress(done / max(total_pages, 1))

    prog.empty()
    df = pd.DataFrame(rows_normed)
    if df.empty:
        return df

    df["norm_name"] = df["Name"].map(_normalize_name)
    df["reg_norm"]  = df["Registration No."].str.replace(r"\s+", "", regex=True).str.upper()
    df.drop_duplicates(subset=["source", "reg_norm", "norm_name"], inplace=True)
    return df[["source", "Registration No.", "Name", "norm_name", "reg_norm"]].reset_index(drop=True)

# ------------------------------------------------------------
# Supabase helpers
# ------------------------------------------------------------
def _get_supabase() -> Optional[Client]:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"].get("service_key") or st.secrets["supabase"]["anon_key"]
        if not create_client:
            return None
        return create_client(url, key)  # type: ignore
    except Exception:
        return None

@st.cache_data(ttl=24 * 60 * 60)
def supabase_read_registry() -> pd.DataFrame:
    sb = _get_supabase()
    if not sb:
        return pd.DataFrame()
    try:
        resp = sb.table("ibbi_registry").select("*").execute()
        data = resp.data or []
        df = pd.DataFrame(data)
        if df.empty:
            return df
        # harmonize
        if "Registration No." not in df.columns and "registration_no" in df.columns:
            df.rename(columns={"registration_no": "Registration No."}, inplace=True)
        if "Name" not in df.columns and "name" in df.columns:
            df.rename(columns={"name": "Name"}, inplace=True)
        if "norm_name" not in df.columns:
            df["norm_name"] = df["Name"].map(_normalize_name)
        if "reg_norm" not in df.columns:
            df["reg_norm"] = df["Registration No."].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
        return df[["source", "Registration No.", "Name", "norm_name", "reg_norm"]]
    except Exception:
        return pd.DataFrame()

def supabase_replace_registry(df: pd.DataFrame) -> bool:
    """Deletes table and re-inserts df. Requires service_key."""
    sb = _get_supabase()
    if not sb:
        return False
    try:
        # Ensure we’re using service key (writes)
        sb.table("ibbi_registry").delete().neq("reg_norm", "").execute()
        # Insert in chunks
        recs = df.to_dict(orient="records")
        chunk = 1000
        for i in range(0, len(recs), chunk):
            sb.table("ibbi_registry").upsert(recs[i:i+chunk], on_conflict="source,reg_norm,norm_name").execute()
        return True
    except Exception:
        return False

# ------------------------------------------------------------
# Business rules
# ------------------------------------------------------------
def check_tenure_leq_4yrs(
    appoint: Optional[pd.Timestamp],
    resign: Optional[pd.Timestamp],
) -> Tuple[bool, Optional[float]]:
    if isinstance(appoint, pd.Timestamp):
        end_dt = resign if isinstance(resign, pd.Timestamp) else pd.Timestamp(date.today())
        yrs = _years_between(appoint, end_dt)
        return yrs <= 4.0, yrs
    return False, None

def check_ibbi_registered(name: str, reg_no: str, registry: pd.DataFrame) -> bool:
    if registry is None or registry.empty:
        return False
    reg_norm = (reg_no or "").replace(" ", "").upper()
    if reg_norm and (registry["reg_norm"] == reg_norm).any():
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

    # --- Sidebar: only Segment + Data URL (as requested) ---
    with st.sidebar:
        seg = st.selectbox("Select Segment", ["REIT", "InvIT"], index=0)
        sheet_url = st.text_input(
            "Data URL (public Google Sheet)",
            value=DEFAULT_SHEET_URL,
            help="Public view link; Sheet1 should contain the valuer rows.",
        )

    if not sheet_url:
        st.warning("Please provide a public Google Sheet URL in the sidebar.")
        return

    # Load valuation rows
    val_df = load_valuers_sheet(sheet_url)
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

    # ---------- Registry source (Supabase first, fallback to scrape) ----------
    registry = supabase_read_registry()
    used_supabase = not registry.empty

    if registry.empty:
        with st.spinner("No Supabase registry found. Building once from IBBI (auto-discover)…"):
            registry = _scrape_ibbi_registry_full()
            if not registry.empty:
                # Try to persist if service_key exists
                if supabase_replace_registry(registry):
                    used_supabase = True

    # Optional maintenance control (NOT in sidebar)
    with st.expander("IBBI Registry maintenance"):
        if used_supabase:
            st.success("Using Supabase cache.")
        else:
            st.warning("Using in-memory registry (Supabase unavailable or empty).")

        if st.button("Full refresh from IBBI now (overwrite Supabase)"):
            df_new = _scrape_ibbi_registry_full()
            if df_new.empty:
                st.error("Could not rebuild registry from IBBI.")
            else:
                ok = supabase_replace_registry(df_new)
                if ok:
                    supabase_read_registry.clear()  # clear 24h read cache
                    st.success("Supabase registry overwritten.")
                else:
                    st.warning("Supabase write failed (service_key missing?). Registry kept in memory for this session.")

    # Evaluate rows
    out_rows: List[Dict[str, Any]] = []
    for _, r in filtered.iterrows():
        name   = str(r.get("Name of Valuer", "")).strip()
        regno  = str(r.get("IBBI Registration No.", "")).strip()
        appoint = r.get("Date of Appointmnet")
        resign  = r.get("Date of Resignation")

        ok_tenure, yrs = check_tenure_leq_4yrs(appoint, resign)
        ok_reg    = check_ibbi_registered(name, regno, registry)

        out_rows.append(
            {
                "Name of REIT": r.get("Name of REIT"),
                "Financial Year": r.get("Finanical Year"),
                "Valuer": name,
                "IBBI Reg No.": regno,
                "Appointment": appoint.date().isoformat() if isinstance(appoint, pd.Timestamp) else "",
                "Resignation": resign.date().isoformat() if isinstance(resign, pd.Timestamp) else "",
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
