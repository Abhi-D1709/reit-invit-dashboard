# tabs/valuation.py
from __future__ import annotations

import os
import re
import math
import concurrent.futures as cf
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ------------------------------------------------------------
# Config & helpers from your common utilities
# ------------------------------------------------------------
try:
    from utils.common import (
        VALUATION_REIT_SHEET_URL,
        inject_global_css,
        load_table_url,
    )
    DEFAULT_VALUATION_URL = VALUATION_REIT_SHEET_URL.strip()
except Exception:
    DEFAULT_VALUATION_URL = ""
    def inject_global_css() -> None: pass

    def _gsheet_csv_from_share(url: str, gid: Optional[int] = None) -> str:
        if not url: return url
        m = re.match(r"(https://docs\.google\.com/spreadsheets/d/[^/]+)", url.strip())
        if not m: return url
        base = m.group(1)
        return f"{base}/export?format=csv" if gid is None else f"{base}/export?format=csv&gid={gid}"

    def load_table_url(url: str, sheet: Optional[str] = None, gid: Optional[int] = None) -> pd.DataFrame:
        csv_url = _gsheet_csv_from_share(url, gid=gid)
        try:
            return pd.read_csv(csv_url)
        except Exception:
            try:
                pub = url.replace("/edit", "/pubhtml")
                tables = pd.read_html(pub)
                return tables[0] if tables else pd.DataFrame()
            except Exception:
                return pd.DataFrame()

# ------------------------------------------------------------
# Supabase client & Fetching Logic
# ------------------------------------------------------------
def _get_supabase_creds() -> Tuple[str, str]:
    url, key = "", ""
    try:
        url = st.secrets.get("supabase_valuation", {}).get("url", "")
        key = st.secrets.get("supabase_valuation", {}).get("anon_key", "")
    except Exception:
        pass

    if not url or not key:
        try:
            url = url or st.secrets.get("supabase", {}).get("url", "")
            key = key or st.secrets.get("supabase", {}).get("anon_key", "")
        except Exception:
            pass

    if not url or not key:
        url = url or os.getenv("SUPABASE_URL", "")
        key = key or os.getenv("SUPABASE_KEY", "")

    return url, key

@st.cache_resource(show_spinner=False)
def _sb_client():
    from supabase import create_client
    url, key = _get_supabase_creds()
    if not url or not key:
        raise RuntimeError("Supabase credentials not found.")
    return create_client(url, key)

# UPDATED: Added pagination to fetch ALL rows, not just the first 1000
@st.cache_data(ttl=3600, show_spinner="Loading registry from database...")
def _sb_select_all(table: str) -> pd.DataFrame:
    all_rows = []
    start = 0
    batch_size = 1000  # Matches default Supabase API limit
    
    client = _sb_client()
    
    while True:
        try:
            # .range() is inclusive: 0-999, 1000-1999
            resp = client.table(table).select("*").range(start, start + batch_size - 1).execute()
            rows = resp.data or []
            
            if not rows:
                break
                
            all_rows.extend(rows)
            
            # If we got fewer rows than requested, we've reached the end
            if len(rows) < batch_size:
                break
                
            start += batch_size
        except Exception as e:
            st.error(f"Error fetching from {table}: {e}")
            break

    df = pd.DataFrame(all_rows)
    if not df.empty:
        if "reg_no" in df.columns:
            df["reg_no"] = df["reg_no"].astype(str).str.strip().str.upper()
        if "name" in df.columns:
            df["name"] = df["name"].astype(str).str.strip()
            
    return df

def _sb_upsert(table: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    client = _sb_client()
    CHUNK = 1000
    for i in range(0, len(rows), CHUNK):
        batch = rows[i : i + CHUNK]
        try:
            client.table(table).upsert(batch, on_conflict="reg_no", ignore_duplicates=False).execute()
        except Exception as e:
            st.warning(f"Failed to upsert batch to {table}: {e}")

# ------------------------------------------------------------
# IBBI registry scraping
# ------------------------------------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml",
    "Connection": "keep-alive",
}
INDIV_BASE = "https://ibbi.gov.in/service-provider/rvs?page={page}"
ENT_BASE   = "https://ibbi.gov.in/service-provider/rvo-entities?page={page}"

def _fetch(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200: return None
        return r.text
    except Exception:
        return None

def _parse_individuals(html: str) -> List[Dict[str, Any]]:
    out = []
    if not html: return out
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="reporttable")
    if not table: return out
    tbody = table.find("tbody")
    if not tbody: return out
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 8: continue
        reg_no = (tds[1] or "").strip().upper()
        if not reg_no: continue
        out.append({
            "reg_no": reg_no,
            "name": (tds[2] or "").strip(),
            "address": (tds[3] or "").strip(),
            "email": (tds[4] or "").strip(),
            "rvo": (tds[5] or "").strip(),
            "registration_date": (tds[6] or "").strip(),
            "asset_class": (tds[7] or "").strip(),
        })
    return out

def _parse_entities(html: str) -> List[Dict[str, Any]]:
    out = []
    if not html: return out
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="reporttable")
    if not table: return out
    tbody = table.find("tbody")
    if not tbody: return out
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 9: continue
        reg_no = (tds[1] or "").strip().upper()
        if not reg_no: continue
        out.append({
            "reg_no": reg_no,
            "name": (tds[3] or "").strip(),
            "constitution": (tds[2] or "").strip(),
            "address": (tds[4] or "").strip(),
            "email": (tds[5] or "").strip(),
            "rvo": (tds[6] or "").strip(),
            "directors": (tds[7] or "").strip(),
            "asset_class": (tds[8] or "").strip(),
        })
    return out

def _has_rows(base_url: str, parse_fn, page: int) -> bool:
    html = _fetch(base_url.format(page=page))
    rows = parse_fn(html) if html else []
    return len(rows) > 0

def _discover_last_page(base_url: str, parse_fn) -> int:
    if not _has_rows(base_url, parse_fn, 1): return 0
    lo, hi = 1, 64
    while _has_rows(base_url, parse_fn, hi):
        lo, hi = hi + 1, hi * 2
        if hi > 4000: break
    left, right, last_with_rows = lo, hi, 1
    while left <= right:
        mid = (left + right) // 2
        if _has_rows(base_url, parse_fn, mid):
            last_with_rows = mid
            left = mid + 1
        else:
            right = mid - 1
    return last_with_rows

def _fetch_all_pages(base_url: str, parse_fn) -> List[Dict[str, Any]]:
    last = _discover_last_page(base_url, parse_fn)
    if last <= 0: return []
    pages = list(range(1, last + 1))
    results = []
    with cf.ThreadPoolExecutor(max_workers=min(32, len(pages) or 1)) as ex:
        futures = {ex.submit(_fetch, base_url.format(page=p)): p for p in pages}
        for fut in cf.as_completed(futures):
            html = fut.result()
            if html: results.extend(parse_fn(html))
    return results

def scrape_ibbi_registry() -> Tuple[pd.DataFrame, pd.DataFrame]:
    indiv_rows = _fetch_all_pages(INDIV_BASE, _parse_individuals)
    ent_rows   = _fetch_all_pages(ENT_BASE, _parse_entities)
    df_ind = pd.DataFrame(indiv_rows)
    df_ent = pd.DataFrame(ent_rows)
    for df in (df_ind, df_ent):
        if not df.empty:
            df["reg_no"] = df["reg_no"].astype(str).str.strip().str.upper()
            if "name" in df.columns:
                df["name"] = df["name"].astype(str).str.strip()
    return df_ind, df_ent

def _ensure_registry(force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Clear cache if forcing refresh
    if force_refresh:
        _sb_select_all.clear()
        
    ind, ent = pd.DataFrame(), pd.DataFrame()
    
    # 2. Try DB first (now supports pagination)
    if not force_refresh:
        ind = _sb_select_all("ibbi_rv_individuals")
        ent = _sb_select_all("ibbi_rv_entities")
        if not ind.empty or not ent.empty:
            return ind, ent

    # 3. Scrape if forced or if DB was empty
    with st.spinner("Fetching latest IBBI registry data from ibbi.gov.in..."):
        ind_new, ent_new = scrape_ibbi_registry()
        
        # 4. Upsert to Supabase
        if not ind_new.empty:
            _sb_upsert("ibbi_rv_individuals", ind_new.to_dict("records"))
        if not ent_new.empty:
            _sb_upsert("ibbi_rv_entities", ent_new.to_dict("records"))
            
        # 5. Clear cache again so next load pulls the fresh data from DB
        _sb_select_all.clear()
        
        # 6. Return the newly scraped data directly for this run
        return ind_new, ent_new

# ------------------------------------------------------------
# Valuation logic
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_valuation_sheet(url: str) -> pd.DataFrame:
    try:
        df = load_table_url(url, sheet="Sheet1")
    except TypeError:
        df = load_table_url(url)

    rename_map = {
        "Finanical Year": "Financial Year",
        "Date of Appointmnet": "Date of Appointment",
        "IBBI Registration No.": "IBBI Registration No",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns: df[dst] = df[src]

    expected = ["Name of REIT", "Financial Year", "Name of Valuer", 
                "Date of Appointment", "Date of Resignation", "IBBI Registration No"]
    keep = [c for c in expected if c in df.columns]
    return df[keep].copy() if keep else pd.DataFrame(columns=expected)

def _parse_date(s: Any) -> Optional[date]:
    if s is None: return None
    t = str(s).strip()
    if not t or t.upper() in {"NA", "N/A", "NONE", "-", "NIL"}: return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"):
        try: return datetime.strptime(t, fmt).date()
        except ValueError: continue
    try:
        d = pd.to_datetime(t, dayfirst=True, errors="coerce")
        if pd.isna(d): return None
        return d.date()
    except Exception: return None

def _fy_end(fy: str) -> Optional[date]:
    m = re.match(r"^\s*(\d{4})\s*[-/]\s*(\d{2})\s*$", str(fy))
    if not m: return None
    start = int(m.group(1))
    return date(start + 1, 3, 31)

def _tenure_days(start_dt: Optional[date], end_dt: Optional[date]) -> Optional[int]:
    if not start_dt or not end_dt: return None
    return (end_dt - start_dt).days

# ------------------------------------------------------------
# Improved Matching Logic
# ------------------------------------------------------------
def _norm_name(s: str) -> str:
    """
    Normalizes names by:
    1. Upper casing
    2. Removing common honorifics (Mr, Ms, Mrs, Dr, etc.)
    3. Removing ALL non-alphanumeric characters (spaces, dots, etc.)
    
    Example: "Mr. Manish Gupta" -> "MANISHGUPTA"
             "L. Anuradha"      -> "LANURADHA"
    """
    if not s: return ""
    
    # 1. Basic cleanup
    text = s.upper().strip()
    
    # 2. Remove common prefixes (honorifics). 
    # We use a regex with \b to ensure we don't cut off real names starting with these letters.
    # e.g., "Mrs." -> removed, but "Mrigank" -> kept.
    prefixes = [
        r"^MR[\.\s]+", r"^MS[\.\s]+", r"^MRS[\.\s]+", r"^DR[\.\s]+", 
        r"^CA[\.\s]+", r"^CS[\.\s]+", r"^CMA[\.\s]+", r"^AR[\.\s]+"
    ]
    for p in prefixes:
        text = re.sub(p, "", text)

    # 3. Remove all non-alphanumeric characters (spaces, dots, commas)
    # This ensures "L. Anuradha" == "L Anuradha" == "LAnuradha"
    return re.sub(r"[^A-Z0-9]", "", text)

def _match_in_registry(reg_no: str, valuer_name: str,
                       ibbi_ind: pd.DataFrame, ibbi_ent: pd.DataFrame) -> Tuple[bool, str]:
    rn = (reg_no or "").strip().upper()
    
    # Normalize the input name from the Google Sheet
    target_name = _norm_name(valuer_name)
    
    # 1. Try matching by Registration Number (Primary Key - Most Accurate)
    if rn:
        if not ibbi_ind.empty and rn in set(ibbi_ind["reg_no"]): return True, "Individual"
        if not ibbi_ent.empty and rn in set(ibbi_ent["reg_no"]): return True, "Entity"

    # 2. Try matching by Name (Normalized)
    if target_name:
        # Check Individuals
        if not ibbi_ind.empty:
            # Create a set of normalized names from the registry ONCE for speed
            ind_names = set(_norm_name(x) for x in ibbi_ind["name"].astype(str))
            if target_name in ind_names:
                return True, "Individual"

        # Check Entities
        if not ibbi_ent.empty:
            ent_names = set(_norm_name(x) for x in ibbi_ent["name"].astype(str))
            if target_name in ent_names:
                return True, "Entity"

    return False, ""

def evaluate_rows(df: pd.DataFrame, ibbi_ind: pd.DataFrame, ibbi_ent: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    out["Appointment Date"] = out["Date of Appointment"].map(_parse_date)
    out["Resignation Date"] = out["Date of Resignation"].map(_parse_date)
    out["FY End"] = out["Financial Year"].map(_fy_end)
    out["Tenure End"] = out.apply(lambda r: r["Resignation Date"] if pd.notna(r["Resignation Date"]) else r["FY End"], axis=1)
    out["Tenure (days)"] = out.apply(lambda r: _tenure_days(r["Appointment Date"], r["Tenure End"]), axis=1)
    out["Tenure (years)"] = out["Tenure (days)"].map(lambda d: round(d / 365.25, 2) if pd.notna(d) else None)
    out["Tenure ≤ 4 years"] = out["Tenure (days)"].map(lambda d: bool(d is not None and d <= 4 * 365.25))

    matches = out.apply(lambda r: _match_in_registry(str(r.get("IBBI Registration No", "")), str(r.get("Name of Valuer", "")), ibbi_ind, ibbi_ent), axis=1)
    out["IBBI Registered?"] = [m[0] for m in matches]
    out["Matched Type"]     = [m[1] for m in matches]
    out["Tenure Status"] = out["Tenure ≤ 4 years"].map(lambda ok: "✅ OK" if ok else "❌ > 4 years")
    out["IBBI Status"]   = out["IBBI Registered?"].map(lambda ok: "✅ Found in registry" if ok else "❌ Not found")
    return out

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
def render():
    st.header("Valuation")
    inject_global_css()

    with st.sidebar:
        st.subheader("Select Segment")
        seg = st.radio("Segment", ["REIT", "InvIT"], index=0, label_visibility="collapsed")
        st.caption("Data Source (Google Sheet – public view)")
        st.code(DEFAULT_VALUATION_URL or "(not set in utils/common.py)", language="text")
        
        st.divider()
        st.subheader("Data Controls")
        force_refresh = st.button("🔄 Refresh IBBI Registry Data", help="Scrape latest data from ibbi.gov.in and update Supabase.")

    if seg != "REIT":
        st.info("Valuation checks for InvIT will be added similarly. Currently enabled for REIT.")
        return

    if not DEFAULT_VALUATION_URL:
        st.error("VALUATION_REIT_SHEET_URL is not configured in utils/common.py")
        return

    df_raw = load_valuation_sheet(DEFAULT_VALUATION_URL)
    if df_raw.empty:
        st.warning("No valuation rows found in Sheet1.")
        return

    # Filters
    c1, c2 = st.columns([2, 1], vertical_alignment="center")
    ent_list = sorted(df_raw["Name of REIT"].dropna().unique().tolist())
    fy_list  = sorted(df_raw["Financial Year"].dropna().unique().tolist())

    entity = c1.selectbox("Entity", ent_list, index=0)
    fy     = c2.selectbox("Financial Year", ["All"] + fy_list, index=0)

    # Fetch registry with pagination
    ibbi_ind, ibbi_ent = _ensure_registry(force_refresh=force_refresh)
    
    if force_refresh:
        st.success(f"Registry updated! Found {len(ibbi_ind)} individuals and {len(ibbi_ent)} entities.")

    # Filter rows
    q = df_raw[df_raw["Name of REIT"] == entity].copy()
    if fy != "All":
        q = q[q["Financial Year"] == fy]

    if q.empty:
        st.info("No rows for the selected filters.")
        return

    # Evaluate
    eval_df = evaluate_rows(q, ibbi_ind, ibbi_ent)

    st.markdown("### Results")
    view_cols = ["Name of REIT", "Financial Year", "Name of Valuer", "IBBI Registration No", 
                 "Date of Appointment", "Date of Resignation", "Tenure (years)", 
                 "Tenure Status", "IBBI Status", "Matched Type"]
    show_cols = [c for c in view_cols if c in eval_df.columns]
    
    st.dataframe(
        eval_df[show_cols].sort_values(["Financial Year", "Name of Valuer"], na_position="last"),
        use_container_width=True,
        hide_index=True
    )

    breaches_tenure = eval_df[~eval_df["Tenure ≤ 4 years"].fillna(True)]
    breaches_ibbi   = eval_df[~eval_df["IBBI Registered?"].fillna(True)]

    if not breaches_tenure.empty or not breaches_ibbi.empty:
        st.markdown("### Alerts")
        if not breaches_tenure.empty:
            st.error(f"Tenure > 4 years: {len(breaches_tenure)} row(s).")
            st.dataframe(breaches_tenure[show_cols], use_container_width=True, hide_index=True)
        if not breaches_ibbi.empty:
            st.error(f"IBBI registration not found: {len(breaches_ibbi)} row(s).")
            st.dataframe(breaches_ibbi[show_cols], use_container_width=True, hide_index=True)

def render_valuation():
    render()