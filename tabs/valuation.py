# tabs/valuation.py

from __future__ import annotations

import concurrent.futures as cf
import math
import re
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# ------------------------------------------------------------
# Config & utilities
# ------------------------------------------------------------
try:
    # Your shared helpers and URL live here
    from utils.common import (
        VALUATION_REIT_SHEET_URL,  # <- you added this in common.py
        inject_global_css,         # keep your global look
    )
    DEFAULT_VALUATION_URL = VALUATION_REIT_SHEET_URL.strip()
except Exception:
    inject_global_css = lambda: None  # no-op fallback
    DEFAULT_VALUATION_URL = ""

# If you already have a load_table_url helper in utils.common, use it.
# Else fall back to a simple Google Sheets CSV-export reader.
try:
    from utils.common import load_table_url  # type: ignore
    _HAS_COMMON_LOADER = True
except Exception:
    _HAS_COMMON_LOADER = False

    def _gsheet_csv_from_share(url: str, gid: Optional[int] = None) -> str:
        """
        Convert a '.../edit?...' public share link to a CSV export link.
        If gid is None, Google serves the active sheet.
        """
        if not url:
            return url
        m = re.match(r"(https://docs\.google\.com/spreadsheets/d/[^/]+)", url.strip())
        if not m:
            return url
        base = m.group(1)
        if gid is None:
            return f"{base}/export?format=csv"
        return f"{base}/export?format=csv&gid={gid}"

    def load_table_url(url: str, sheet: Optional[str] = None, gid: Optional[int] = None) -> pd.DataFrame:
        """
        Lightweight fallback loader: reads public Google Sheet via CSV export.
        If 'sheet' is provided but gid is unknown, we still try CSV (active sheet).
        """
        csv_url = _gsheet_csv_from_share(url, gid=gid)
        try:
            return pd.read_csv(csv_url)
        except Exception:
            # final fallback: try pandas read_html on 'pubhtml' (requires Publish to web)
            pub = url.replace("/edit", "/pubhtml")
            try:
                tables = pd.read_html(pub)
                return tables[0] if tables else pd.DataFrame()
            except Exception:
                return pd.DataFrame()

# ------------------------------------------------------------
# Supabase (anon key only, like your trading module)
# ------------------------------------------------------------
from supabase import create_client

@st.cache_resource(show_spinner=False)
def sb():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["anon_key"]  # anon only
    return create_client(url, key)

def _sb_load_registry(table: str) -> pd.DataFrame:
    """
    Load entire registry table. Returns empty DF if table missing/empty.
    """
    try:
        resp = sb().table(table).select("*").execute()
        data = resp.data or []
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # normalize basic columns we care about
        if "reg_no" in df.columns:
            df["reg_no"] = df["reg_no"].astype(str).str.strip().str.upper()
        if "name" in df.columns:
            df["name"] = df["name"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()

def _sb_registry_is_empty() -> bool:
    ind = _sb_load_registry("ibbi_rv_individuals")
    ent = _sb_load_registry("ibbi_rv_entities")
    return ind.empty and ent.empty

def _sb_upsert(table: str, rows: List[Dict]):
    """
    Upsert rows with on_conflict 'reg_no'. Requires a UNIQUE index on reg_no and
    permissive RLS (select/insert/update) for anon.
    """
    if not rows:
        return
    # chunk upserts to keep payloads reasonable
    client = sb()
    CHUNK = 1000
    for i in range(0, len(rows), CHUNK):
        batch = rows[i : i + CHUNK]
        client.table(table).upsert(
            batch, on_conflict="reg_no", ignore_duplicates=False
        ).execute()

# ------------------------------------------------------------
# IBBI registry scraping (auto-discover page count, concurrent)
# ------------------------------------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

INDIV_BASE = "https://ibbi.gov.in/service-provider/rvs?page={page}"
ENT_BASE   = "https://ibbi.gov.in/service-provider/rvo-entities?page={page}"

def _fetch_page(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None

def _parse_individuals(html: str) -> List[Dict]:
    """
    Parse Individual RV table rows.
    Expected table with class 'reporttable' and 'Registration No.' column.
    """
    out: List[Dict] = []
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="reporttable")
    if not table:
        return out
    tbody = table.find("tbody")
    if not tbody:
        return out
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 8:
            continue
        reg_no = (tds[1] or "").strip().upper()
        name   = (tds[2] or "").strip()
        addr   = (tds[3] or "").strip()
        email  = (tds[4] or "").strip()
        rvo    = (tds[5] or "").strip()
        doj    = (tds[6] or "").strip()
        asset  = (tds[7] or "").strip()
        if not reg_no:
            continue
        out.append({
            "reg_no": reg_no,
            "name": name,
            "address": addr,
            "email": email,
            "rvo": rvo,
            "registration_date": doj,
            "asset_class": asset,
        })
    return out

def _parse_entities(html: str) -> List[Dict]:
    """
    Parse Entity RV table rows.
    """
    out: List[Dict] = []
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="reporttable")
    if not table:
        return out
    tbody = table.find("tbody")
    if not tbody:
        return out
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 9:
            continue
        reg_no = (tds[1] or "").strip().upper()
        constitution = (tds[2] or "").strip()
        name = (tds[3] or "").strip()
        addr = (tds[4] or "").strip()
        email = (tds[5] or "").strip()
        rvo = (tds[6] or "").strip()
        directors = (tds[7] or "").strip()
        asset = (tds[8] or "").strip()
        if not reg_no:
            continue
        out.append({
            "reg_no": reg_no,
            "name": name,
            "constitution": constitution,
            "address": addr,
            "email": email,
            "rvo": rvo,
            "directors": directors,
            "asset_class": asset,
        })
    return out

def _discover_and_fetch(base_url: str, parse_fn, max_step: int = 64) -> List[Dict]:
    """
    Quickly discover how many pages exist by probing in exponentially growing chunks,
    then fetch the discovered range concurrently.
    """
    results: List[Dict] = []

    # Phase 1: discover upper bound of pages that still return rows
    lo, hi = 1, max_step
    def has_rows(page: int) -> bool:
        html = _fetch_page(base_url.format(page=page))
        rows = parse_fn(html) if html else []
        return len(rows) > 0

    # Ensure page 1 exists (if not, nothing to do)
    if not has_rows(1):
        return results

    # Expand upper bound until we hit an empty page
    while has_rows(hi):
        lo = hi + 1
        hi *= 2
        if hi > 4000:
            break  # sanity

    # Binary search to find last page with rows
    left, right = lo, hi
    last_with_rows = 1
    while left <= right:
        mid = (left + right) // 2
        if has_rows(mid):
            last_with_rows = mid
            left = mid + 1
        else:
            right = mid - 1

    # Phase 2: fetch all pages 1..last_with_rows concurrently
    pages = list(range(1, last_with_rows + 1))
    with cf.ThreadPoolExecutor(max_workers=min(32, len(pages) or 1)) as ex:
        futures = {ex.submit(_fetch_page, base_url.format(page=p)): p for p in pages}
        for fut in cf.as_completed(futures):
            html = fut.result()
            if not html:
                continue
            rows = parse_fn(html)
            results.extend(rows)

    return results

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def scrape_ibbi_registry() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scrape both registries, return (individuals_df, entities_df).
    Cached 7 days to avoid accidental re-runs if Supabase is empty.
    """
    indiv_rows = _discover_and_fetch(INDIV_BASE, _parse_individuals)
    ent_rows   = _discover_and_fetch(ENT_BASE, _parse_entities)

    df_ind = pd.DataFrame(indiv_rows)
    df_ent = pd.DataFrame(ent_rows)

    # Normalize keys
    if not df_ind.empty:
        df_ind["reg_no"] = df_ind["reg_no"].astype(str).str.strip().str.upper()
        df_ind["name"]   = df_ind["name"].astype(str).str.strip()
    if not df_ent.empty:
        df_ent["reg_no"] = df_ent["reg_no"].astype(str).str.strip().str.upper()
        df_ent["name"]   = df_ent["name"].astype(str).str.strip()

    return df_ind, df_ent

def ensure_registry_available() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Try Supabase first; if empty, scrape once and upsert to Supabase.
    """
    ind = _sb_load_registry("ibbi_rv_individuals")
    ent = _sb_load_registry("ibbi_rv_entities")
    if not ind.empty or not ent.empty:
        return ind, ent

    # One-time scrape
    with st.spinner("Building local IBBI registry cache (first run)…"):
        ind, ent = scrape_ibbi_registry()
        if not ind.empty:
            _sb_upsert("ibbi_rv_individuals", ind.to_dict("records"))
        if not ent.empty:
            _sb_upsert("ibbi_rv_entities", ent.to_dict("records"))
    return ind, ent

# ------------------------------------------------------------
# Valuation sheet load + business rules
# ------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_valuation_sheet(url: str) -> pd.DataFrame:
    # Your sheet says data is on Sheet1
    try:
        df = load_table_url(url, sheet="Sheet1")
    except TypeError:
        # if your load_table_url has a different signature, try without name
        df = load_table_url(url)

    # Standardize columns expected:
    # Name of REIT | Finanical Year | Name of Valuer | Date of Appointmnet | Date of Resignation | IBBI Registration No.
    # Fix common typos
    rename_map = {
        "Finanical Year": "Financial Year",
        "Date of Appointmnet": "Date of Appointment",
        "IBBI Registration No.": "IBBI Registration No",
    }
    for a, b in rename_map.items():
        if a in df.columns and b not in df.columns:
            df[b] = df[a]

    expected = [
        "Name of REIT",
        "Financial Year",
        "Name of Valuer",
        "Date of Appointment",
        "Date of Resignation",
        "IBBI Registration No",
    ]
    keep = [c for c in expected if c in df.columns]
    return df[keep].copy() if keep else pd.DataFrame(columns=expected)

def _parse_date(s: object) -> Optional[date]:
    if s is None:
        return None
    t = str(s).strip()
    if not t or t.upper() in {"NA", "N/A", "NONE", "-"}:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    # last attempt: pandas
    try:
        d = pd.to_datetime(t, dayfirst=True, errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None

def _fy_end(fy: str) -> Optional[date]:
    """
    "2024-25" -> 31-Mar-2025 ; "2019-20" -> 31-Mar-2020
    """
    m = re.match(r"^\s*(\d{4})\s*[-/]\s*(\d{2})\s*$", str(fy).strip())
    if not m:
        return None
    start = int(m.group(1))
    end_year = start + 1
    return date(end_year, 3, 31)

def _tenure_days(start_dt: Optional[date], end_dt: Optional[date]) -> Optional[int]:
    if not start_dt or not end_dt:
        return None
    return (end_dt - start_dt).days

def _normalize_name(s: str) -> str:
    return re.sub(r"[\s\W]+", " ", (s or "")).strip().upper()

def match_in_registry(reg_no: str, valuer_name: str,
                      ind: pd.DataFrame, ent: pd.DataFrame) -> Tuple[bool, str]:
    """
    Try to match first by reg_no; if not present, fallback to name match.
    Returns (is_registered, matched_kind: 'Individual'|'Entity'|'' )
    """
    rn = (reg_no or "").strip().upper()
    nm = _normalize_name(valuer_name)

    if rn:
        if not ind.empty and rn in set(ind["reg_no"]):
            return True, "Individual"
        if not ent.empty and rn in set(ent["reg_no"]):
            return True, "Entity"

    if nm:
        if not ind.empty and nm in set(_normalize_name(n) for n in ind["name"].astype(str).tolist()):
            return True, "Individual"
        if not ent.empty and nm in set(_normalize_name(n) for n in ent["name"].astype(str).tolist()):
            return True, "Entity"

    return False, ""

def evaluate_rows(df: pd.DataFrame,
                  ibbi_ind: pd.DataFrame,
                  ibbi_ent: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
      - tenure_end (resignation if present else FY end)
      - tenure_days / tenure_years
      - tenure_ok (≤ 4 years)
      - ibbi_registered (match in individuals or entities)
    """
    if df.empty:
        return df

    out = df.copy()
    out["Appointment Date"] = out["Date of Appointment"].map(_parse_date)
    out["Resignation Date"] = out["Date of Resignation"].map(_parse_date)

    # tenure end = resignation if available, else FY end
    out["FY End"] = out["Financial Year"].map(_fy_end)
    out["Tenure End"] = out.apply(
        lambda r: r["Resignation Date"] if pd.notna(r["Resignation Date"]) else r["FY End"],
        axis=1
    )

    out["Tenure (days)"] = out.apply(
        lambda r: _tenure_days(r["Appointment Date"], r["Tenure End"]), axis=1
    )
    out["Tenure (years)"] = out["Tenure (days)"].map(lambda d: round(d / 365.25, 2) if pd.notna(d) else None)
    out["Tenure ≤ 4 years"] = out["Tenure (days)"].map(lambda d: bool(d is not None and d <= 4 * 365.25))

    def _ibbi(row) -> Tuple[bool, str]:
        return match_in_registry(
            str(row.get("IBBI Registration No", "")),
            str(row.get("Name of Valuer", "")),
            ibbi_ind, ibbi_ent
        )

    reg = out.apply(_ibbi, axis=1)
    out["IBBI Registered?"] = [t[0] for t in reg]
    out["Matched Type"] = [t[1] for t in reg]

    # Friendly statuses
    out["Tenure Status"] = out["Tenure ≤ 4 years"].map(lambda x: "✅ OK" if x else "❌ > 4 years")
    out["IBBI Status"]   = out["IBBI Registered?"].map(lambda x: "✅ Found in registry" if x else "❌ Not found")

    return out

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

def render():
    st.header("Valuation")

    inject_global_css()

    # Sidebar: Segment & data URL only (clean)
    with st.sidebar:
        st.subheader("Select Segment")
        seg = st.radio("Segment", ["REIT", "InvIT"], index=0, label_visibility="collapsed")
        st.caption("Data Source (Google Sheet – public view)")
        st.code(DEFAULT_VALUATION_URL or "(not set in utils/common.py)", language="text")

    if seg != "REIT":
        st.info("Valuation checks for InvIT will be added similarly. Currently enabled for REIT.")
        return

    # Load valuation sheet
    if not DEFAULT_VALUATION_URL:
        st.error("VALUATION_REIT_SHEET_URL is not configured in utils/common.py")
        return

    df_raw = load_valuation_sheet(DEFAULT_VALUATION_URL)
    if df_raw.empty:
        st.warning("No valuation rows found in Sheet1. Please check columns and sharing.")
        return

    # Entity & FY selectors (main area)
    cols = st.columns([2, 1, 1, 1])
    ent_list = sorted(df_raw["Name of REIT"].dropna().unique().tolist())
    fy_list  = sorted(df_raw["Financial Year"].dropna().unique().tolist())

    entity = cols[0].selectbox("Entity", ent_list, index=0)
    fy     = cols[1].selectbox("Financial Year", ["All"] + fy_list, index=0)

    # Minimal maintenance (not in sidebar): refresh registry if you need to
    with st.expander("Maintenance (optional)"):
        st.markdown(
            "- Registry source: IBBI (Individuals & Entities). "
            "App reads from Supabase first; if empty, it will scrape once and cache/upsert."
        )
        run_refresh = st.button("Force refresh IBBI registry now (scrape & upsert)")
        if run_refresh:
            ind, ent = scrape_ibbi_registry()
            if not ind.empty:
                _sb_upsert("ibbi_rv_individuals", ind.to_dict("records"))
            if not ent.empty:
                _sb_upsert("ibbi_rv_entities", ent.to_dict("records"))
            st.success(f"Refreshed: {len(ind)} individual rows, {len(ent)} entity rows.")

    # Get registry (Supabase first; if empty, scrape once)
    ibbi_ind, ibbi_ent = ensure_registry_available()

    # Filter the valuation rows
    q = df_raw[df_raw["Name of REIT"] == entity].copy()
    if fy != "All":
        q = q[q["Financial Year"] == fy]

    if q.empty:
        st.info("No rows for the selected filters.")
        return

    # Evaluate checks
    eval_df = evaluate_rows(q, ibbi_ind, ibbi_ent)

    # Present results
    st.markdown("### Results")
    base_cols = [
        "Name of REIT",
        "Financial Year",
        "Name of Valuer",
        "IBBI Registration No",
        "Date of Appointment",
        "Date of Resignation",
        "Tenure (years)",
        "Tenure Status",
        "IBBI Status",
        "Matched Type",
    ]

    # Show a compact summary table
    show_cols = [c for c in base_cols if c in eval_df.columns]
    st.dataframe(
        eval_df[show_cols].sort_values(["Financial Year", "Name of Valuer"], na_position="last"),
        use_container_width=True,
        hide_index=True
    )

    # Alerts (red) for any breaches
    bre_tenure = eval_df[~eval_df["Tenure ≤ 4 years"].fillna(True)]
    bre_ibbi   = eval_df[~eval_df["IBBI Registered?"].fillna(True)]

    if not bre_tenure.empty or not bre_ibbi.empty:
        st.markdown("### Alerts")
    if not bre_tenure.empty:
        st.error(f"Tenure > 4 years: {len(bre_tenure)} row(s).")
        st.dataframe(bre_tenure[show_cols], use_container_width=True, hide_index=True)
    if not bre_ibbi.empty:
        st.error(f"IBBI registration not found: {len(bre_ibbi)} row(s).")
        st.dataframe(bre_ibbi[show_cols], use_container_width=True, hide_index=True)

# Streamlit page entry point (if imported by pages/7_Valuation.py)
def render_valuation():
    render()
