# tabs/valuation.py
from __future__ import annotations

import re
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils import periods, rules, status  # every threshold lives in utils/rules.py
from utils.status import CheckResult, Status

# ------------------------------------------------------------
# Config & helpers from your common utilities
# ------------------------------------------------------------
from utils.common import (
    VALUATION_REIT_SHEET_URL,
    DEFAULT_REIT_FUND_URL,
    DEFAULT_INVIT_FUND_URL,
    ENT_COL,
    load_table_url,
    _standardize_selector_columns,
    _find_col,
)

DEFAULT_VALUATION_URL = VALUATION_REIT_SHEET_URL.strip()
# gid of the "timelines" tab (Sheet2) in the valuation workbook
VALUATION_TIMELINE_GID = "122761239"


# ------------------------------------------------------------
# IBBI registry: scraped weekly by jobs/ingest_ibbi.py, read from the data branch
# ------------------------------------------------------------
from utils.datastore import DataUnavailable, load_ibbi  # noqa: E402


def _load_registry() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """(individuals, entities, meta). meta is None when the registry could not be read."""
    try:
        ind, ent, meta = load_ibbi()
        return ind, ent, meta
    except (DataUnavailable, FileNotFoundError, KeyError, ValueError) as e:
        st.error(f"IBBI registry unavailable, so registration checks are skipped: {e}")
        return pd.DataFrame(), pd.DataFrame(), None


# ------------------------------------------------------------
# Valuation (Sheet 1) Logic
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_valuation_sheet(url: str) -> pd.DataFrame:
    try:
        # Load Sheet1 (Default)
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

def _norm_name(s: str) -> str:
    if not s: return ""
    text = s.upper().strip()
    prefixes = [
        r"^MR[\.\s]+", r"^MS[\.\s]+", r"^MRS[\.\s]+", r"^DR[\.\s]+", 
        r"^CA[\.\s]+", r"^CS[\.\s]+", r"^CMA[\.\s]+", r"^AR[\.\s]+"
    ]
    for p in prefixes:
        text = re.sub(p, "", text)
    return re.sub(r"[^A-Z0-9]", "", text)

def _registry_index(ibbi_ind: pd.DataFrame, ibbi_ent: pd.DataFrame):
    """Lookups by registration number and by normalised name. For each key the entry is
    (type, status); a currently registered row wins over a cancelled one."""
    by_reg: Dict[str, Tuple[str, str]] = {}
    by_name: Dict[str, Tuple[str, str]] = {}

    def add(index: Dict[str, Tuple[str, str]], key: str, kind: str, status: str) -> None:
        if key and (key not in index or (index[key][1] != "Registered" and status == "Registered")):
            index[key] = (kind, status)

    for df, kind in ((ibbi_ind, "Individual"), (ibbi_ent, "Entity")):
        if df.empty:
            continue
        for reg, name, status in zip(df["reg_no"], df["name"], df["status"]):
            add(by_reg, str(reg).strip().upper(), kind, status)
            add(by_name, _norm_name(str(name)), kind, status)
    return by_reg, by_name


def _match_in_registry(reg_no: str, valuer_name: str, index) -> Tuple[Optional[bool], str, str]:
    """(currently registered?, matched type, status note). Registration number is tried
    first, then name. Returns (None, "", "") when there is no registry to check against."""
    by_reg, by_name = index
    if not by_reg and not by_name:
        return None, "", ""
    hit = by_reg.get((reg_no or "").strip().upper()) or by_name.get(_norm_name(valuer_name))
    if not hit:
        return False, "", ""
    kind, status = hit
    return status == "Registered", kind, "" if status == "Registered" else status


def evaluate_rows(df: pd.DataFrame, ibbi_ind: pd.DataFrame, ibbi_ent: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    out["Appointment Date"] = out["Date of Appointment"].map(_parse_date)
    out["Resignation Date"] = out["Date of Resignation"].map(_parse_date)
    out["FY End"] = out["Financial Year"].map(_fy_end)
    out["Tenure End"] = out.apply(lambda r: r["Resignation Date"] if pd.notna(r["Resignation Date"]) else r["FY End"], axis=1)
    out["Tenure (days)"] = out.apply(lambda r: _tenure_days(r["Appointment Date"], r["Tenure End"]), axis=1)
    out["Tenure (years)"] = out["Tenure (days)"].map(lambda d: round(d / 365.25, 2) if pd.notna(d) else None)
    # <NA> when the tenure can't be worked out (missing appointment date or unreadable FY): "insufficient data",
    # not a failure. It used to be False, which reported "> 4 years" for every row with a missing date.
    days = pd.to_numeric(out["Tenure (days)"], errors="coerce")
    out["Tenure within limit"] = (days <= rules.VALUER_MAX_TENURE_YEARS * 365.25).astype("boolean").mask(days.isna())

    index = _registry_index(ibbi_ind, ibbi_ent)
    matches = [
        _match_in_registry(str(r.get("IBBI Registration No", "")), str(r.get("Name of Valuer", "")), index)
        for _, r in out.iterrows()
    ]
    out["IBBI Registered?"] = pd.array([m[0] for m in matches], dtype="boolean")  # <NA> = registry unavailable
    out["Matched Type"] = [m[1] for m in matches]
    out["Tenure Status"] = out["Tenure within limit"].map(
        lambda ok: status.tag(Status.NO_DATA, "Insufficient data") if pd.isna(ok) else (
            status.tag(Status.PASS, "OK") if ok else status.tag(Status.FAIL, f"> {rules.VALUER_MAX_TENURE_YEARS} years"))
    )

    def ibbi_status(m):
        if m[0] is None:
            return status.tag(Status.REVIEW, "Registry unavailable")
        if m[0]:
            return status.tag(Status.PASS, "Found in registry")
        return status.tag(Status.FAIL, m[2] or "Not found")

    out["IBBI Status"] = [ibbi_status(m) for m in matches]
    return out

# ------------------------------------------------------------
# Compliance Logic (Sheet 2)
# ------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_valuation_timelines_sheet(url: str, gid: str) -> pd.DataFrame:
    if "docs.google.com" in url:
        base = re.sub(r"/edit.*", "", url).strip()
        csv_url = f"{base}/export?format=csv&gid={gid}"
        try:
            df = pd.read_csv(csv_url)
            return df
        except Exception:
            return pd.DataFrame()
    return load_table_url(url, gid=int(gid))

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_fundraising_data(url: str) -> pd.DataFrame:
    df = load_table_url(url)
    df = _standardize_selector_columns(df)
    date_col = _find_col(df.columns, aliases=["Date of Fund raising", "Date of Fund Raising"])
    if date_col:
        df["FundDate"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    return df

def check_timelines_and_completeness(df: pd.DataFrame, fund_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    out = df.copy()
    
    col_report = "Date of valuation report from valuer"
    col_trustee = "Date of submission of Valuation Report to Trustee"
    col_nav = "Date of disclosure of NAV to the Stock Exchanges"
    
    # Handle spelling variation in "Disclosure"
    col_discl = "Date of Discloure of valuation report to the stock exchanges"
    if col_discl not in out.columns:
        # Try correct spelling
        col_discl = "Date of Disclosure of valuation report to the stock exchanges"

    # Parse dates
    for c in [col_report, col_trustee, col_nav, col_discl]:
        if c in out.columns:
            out[c + "_dt"] = pd.to_datetime(out[c], dayfirst=True, errors="coerce")
    
    # --- Checks 1, 2, 3: Timelines (15 days) ---
    def calc_delay(row, start_col, end_col, label):
        s = row.get(start_col + "_dt")
        e = row.get(end_col + "_dt")
        if pd.notna(s) and pd.notna(e):
            diff = (e - s).days
            if diff > rules.VALUATION_REPORT_MAX_DAYS:
                return status.tag(Status.FAIL, f"{diff} days ({label})")
            return status.tag(Status.PASS, f"{diff} days")  # Show days even for pass
        return "-"

    if col_report + "_dt" in out.columns:
        if col_trustee + "_dt" in out.columns:
            out["Check: Trustee Submission"] = out.apply(lambda r: calc_delay(r, col_report, col_trustee, "Trustee"), axis=1)
        if col_nav + "_dt" in out.columns:
            out["Check: NAV Disclosure"] = out.apply(lambda r: calc_delay(r, col_report, col_nav, "NAV"), axis=1)
        if col_discl + "_dt" in out.columns:
            out["Check: Report Disclosure"] = out.apply(lambda r: calc_delay(r, col_report, col_discl, "Report"), axis=1)

    # --- Checks 4 & 5: Frequency Completeness ---
    grouped = out.groupby(["Name of REIT", "Financial Year"])
    freq_alerts = []
    
    for (reit, fy), group in grouped:
        has_annual = False
        if "Frequency" in group.columns and group["Frequency"].str.contains("Annual", case=False, na=False).any(): has_annual = True
        if "Period Ended" in group.columns and group["Period Ended"].str.contains("Mar", case=False, na=False).any(): has_annual = True
        
        has_half = False
        if "Frequency" in group.columns and group["Frequency"].str.contains("Half", case=False, na=False).any(): has_half = True
        if "Period Ended" in group.columns and (group["Period Ended"].str.contains("Sept", case=False, na=False).any() or group["Period Ended"].str.contains("Sep", case=False, na=False).any()): has_half = True
        
        if not has_annual:
            freq_alerts.append({"Name of REIT": reit, "Financial Year": fy, "Issue": "Missing Annual/March Valuation"})
        if not has_half:
            freq_alerts.append({"Name of REIT": reit, "Financial Year": fy, "Issue": "Missing Half-Year/Sept Valuation"})
    
    df_freq_alerts = pd.DataFrame(freq_alerts)

    # --- Check 6: Fundraising Correlation ---
    fund_checks = []
    if not fund_df.empty and col_report + "_dt" in out.columns and "Name of REIT" in out.columns:
        type_col = _find_col(fund_df.columns, aliases=["Type of Issue"])
        
        if type_col and "FundDate" in fund_df.columns:
            post_ipo_fund = fund_df[~fund_df[type_col].astype(str).str.contains("Initial", case=False, na=False)].copy()
            
            for idx, f_row in post_ipo_fund.iterrows():
                f_date = f_row["FundDate"]
                if pd.isna(f_date): continue
                
                f_entity = str(f_row.get(ENT_COL, "")).strip()
                val_rows = out[out["Name of REIT"].apply(_norm_name) == _norm_name(f_entity)]
                
                if val_rows.empty:
                    val_rows = out[out["Name of REIT"].str.contains(f_entity[:10], case=False, na=False)]

                # 6-month window logic
                start_window = f_date - timedelta(days=rules.VALUATION_BEFORE_FUNDRAISING_DAYS)
                
                # Find the LATEST valid valuation report in that window
                valid_vals = val_rows[
                    (val_rows[col_report + "_dt"] >= start_window) & 
                    (val_rows[col_report + "_dt"] <= f_date)
                ].sort_values(col_report + "_dt", ascending=False)
                
                verdict = status.tag(Status.FAIL)
                last_val_date = None
                days_prior = None
                
                if not valid_vals.empty:
                    verdict = status.tag(Status.PASS)
                    last_val_date = valid_vals.iloc[0][col_report + "_dt"]
                    days_prior = (f_date - last_val_date).days
                
                fund_checks.append({
                    "Name of REIT": f_entity,
                    "Fundraising Date": f_date.date(),
                    "Issue Type": f_row.get(type_col, "-"),
                    "Latest Valuation Date": last_val_date.date() if last_val_date else "Not Found",
                    "Days Prior": days_prior if days_prior is not None else "-",
                    "Status": verdict
                })
    
    df_fund_checks = pd.DataFrame(fund_checks)

    return out, df_freq_alerts, df_fund_checks

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
AREA = "Valuation"


def summary_results(entity: str) -> list:
    """Scorecard verdicts for the valuers of a REIT's latest financial year: tenure and IBBI registration."""
    df = load_valuation_sheet(DEFAULT_VALUATION_URL)
    rows = df[df["Name of REIT"] == entity] if not df.empty else df
    fy = periods.latest_fy(rows["Financial Year"].dropna().astype(str)) if not rows.empty else None
    if fy is None:
        return [CheckResult("Valuers", Status.NO_DATA, "No valuer rows for this entity", AREA)]
    try:
        ind, ent, _ = load_ibbi()
    except (DataUnavailable, FileNotFoundError, KeyError, ValueError):
        ind = ent = pd.DataFrame(columns=["reg_no", "name", "status"])
    out = evaluate_rows(rows[rows["Financial Year"].astype(str) == fy], ind, ent)

    tenure = out["Tenure within limit"]
    if (tenure == False).any():  # noqa: E712
        t_status, t_msg = Status.FAIL, f"a valuer has been in place more than {rules.VALUER_MAX_TENURE_YEARS} years"
    elif tenure.isna().any():
        t_status, t_msg = Status.NO_DATA, "appointment date missing for a valuer"
    else:
        t_status, t_msg = Status.PASS, "all valuers within the tenure limit"
    reg = out["IBBI Registered?"]
    if (reg == False).any():  # noqa: E712
        r_status, r_msg = Status.FAIL, "a valuer is not currently registered with IBBI"
    elif reg.isna().any():
        r_status, r_msg = Status.REVIEW, "IBBI registry not available"
    else:
        r_status, r_msg = Status.PASS, "all valuers found in the IBBI registry"
    return [
        CheckResult(f"Valuer tenure ≤ {rules.VALUER_MAX_TENURE_YEARS} years", t_status, f"FY {fy}: {t_msg}", AREA, "valuation.tenure_years"),
        CheckResult("Valuer registered with IBBI", r_status, f"FY {fy}: {r_msg}", AREA),
    ]


def render():
    st.header("Valuation")

    with st.sidebar:
        st.subheader("Select Segment")
        seg = st.radio("Segment", ["REIT", "InvIT"], index=0, label_visibility="collapsed")

    if seg != "REIT":
        st.info("Valuation checks for InvIT will be added similarly. Currently enabled for REIT.")
        return

    if not DEFAULT_VALUATION_URL:
        st.error("VALUATION_REIT_SHEET_URL is not configured in utils/common.py")
        return

    # 1. Load Data Upfront to generate Sidebar Options
    df_raw = load_valuation_sheet(DEFAULT_VALUATION_URL)
    df_time = load_valuation_timelines_sheet(DEFAULT_VALUATION_URL, VALUATION_TIMELINE_GID)
    df_fund = load_fundraising_data(DEFAULT_REIT_FUND_URL)

    # 2. Sidebar Filters (Shared)
    with st.sidebar:
        st.divider()
        # Combine unique Entities and FYs from both sheets
        ents_1 = set(df_raw["Name of REIT"].dropna()) if not df_raw.empty else set()
        ents_2 = set(df_time["Name of REIT"].dropna()) if not df_time.empty else set()
        all_ents = sorted(list(ents_1 | ents_2))

        fys_1 = set(df_raw["Financial Year"].dropna()) if not df_raw.empty else set()
        fys_2 = set(df_time["Financial Year"].dropna()) if not df_time.empty else set()
        all_fys = sorted(list(fys_1 | fys_2))

        selected_entity = st.selectbox("Entity", ["All"] + all_ents, key="val_side_ent")
        selected_fy = st.selectbox("Financial Year", ["All"] + all_fys, key="val_side_fy")

    tab_registry, tab_compliance = st.tabs(["Valuer Details & Tenure", "Timelines & Compliance"])

    # ========================== TAB 1: Valuer Details ==========================
    with tab_registry:
        if df_raw.empty:
            st.warning("No valuation rows found in Sheet1.")
        else:
            ibbi_ind, ibbi_ent, ibbi_meta = _load_registry()
            if ibbi_meta:
                st.caption(
                    f"IBBI registry as of {ibbi_meta['generated_at'][:10]} "
                    f"({ibbi_meta['individuals']:,} individuals, {ibbi_meta['entities']:,} entities; refreshed weekly)"
                )

            q = df_raw.copy()
            if selected_entity != "All":
                q = q[q["Name of REIT"] == selected_entity]
            if selected_fy != "All":
                q = q[q["Financial Year"] == selected_fy]

            if q.empty:
                st.info("No rows for the selected filters.")
            else:
                eval_df = evaluate_rows(q, ibbi_ind, ibbi_ent)
                st.markdown("### Results")
                view_cols = ["Name of REIT", "Financial Year", "Name of Valuer", "IBBI Registration No", 
                             "Date of Appointment", "Date of Resignation", "Tenure (years)", 
                             "Tenure Status", "IBBI Status", "Matched Type"]
                show_cols = [c for c in view_cols if c in eval_df.columns]
                st.dataframe(eval_df[show_cols].sort_values(["Financial Year", "Name of Valuer"], na_position="last"), width="stretch", hide_index=True)

                breaches_tenure = eval_df[~eval_df["Tenure within limit"].fillna(True)]
                breaches_ibbi   = eval_df[~eval_df["IBBI Registered?"].fillna(True)]
                if not breaches_tenure.empty or not breaches_ibbi.empty:
                    st.markdown("### Alerts")
                    if not breaches_tenure.empty:
                        st.error(f"Tenure > {rules.VALUER_MAX_TENURE_YEARS} years: {len(breaches_tenure)} row(s).")
                        st.dataframe(breaches_tenure[show_cols], width="stretch", hide_index=True)
                    if not breaches_ibbi.empty:
                        st.error(f"IBBI registration not found or cancelled: {len(breaches_ibbi)} row(s).")
                        st.dataframe(breaches_ibbi[show_cols], width="stretch", hide_index=True)

    # ========================== TAB 2: Timelines & Compliance ==========================
    with tab_compliance:
        if df_time.empty:
            st.warning("Could not load Sheet2 (Timelines Data). Please check the Sheet GID.")
        else:
            q_time = df_time.copy()
            if selected_entity != "All": 
                q_time = q_time[q_time["Name of REIT"] == selected_entity]
            if selected_fy != "All": 
                q_time = q_time[q_time["Financial Year"] == selected_fy]

            checked_df, freq_alerts, fund_checks = check_timelines_and_completeness(q_time, df_fund)
            
            st.subheader(f"1. Submission & Disclosure Timelines (Max {rules.VALUATION_REPORT_MAX_DAYS} days)")
            st.caption("Includes Check 3: Date of valuation report vs Date of Disclosure to stock exchanges.")
            
            base_cols = ["Name of REIT", "Financial Year", "Frequency", "Period Ended", "Date of valuation report from valuer"]
            check_cols = [c for c in checked_df.columns if c.startswith("Check:")]
            
            if not checked_df.empty:
                st.dataframe(checked_df[base_cols + check_cols], width="stretch", hide_index=True)
                
                err_mask = False
                for c in check_cols: err_mask |= checked_df[c].astype(str).str.startswith(status.GLYPH[Status.FAIL])
                timeline_errors = checked_df[err_mask]
                if not timeline_errors.empty:
                    st.error(f"Found {len(timeline_errors)} timeline violations.")
                    st.dataframe(timeline_errors[base_cols + check_cols], width="stretch", hide_index=True)
            else:
                st.info("No data for timeline checks.")

            st.divider()

            st.subheader("2. Valuation Frequency Checks")
            if not freq_alerts.empty:
                f_alerts_show = freq_alerts.copy()
                if selected_entity != "All": f_alerts_show = f_alerts_show[f_alerts_show["Name of REIT"] == selected_entity]
                if selected_fy != "All": f_alerts_show = f_alerts_show[f_alerts_show["Financial Year"] == selected_fy]
                if not f_alerts_show.empty:
                    st.error(f"Found {len(f_alerts_show)} missing valuation reports.")
                    st.dataframe(f_alerts_show, width="stretch", hide_index=True)
                else:
                    st.success("All required frequencies found for selection.")
            else:
                st.success("All required frequencies found.")

            st.divider()

            st.subheader("3. Fundraising vs. Valuation")
            st.caption(f"Proof Table: Checking for a valuation report within {rules.VALUATION_BEFORE_FUNDRAISING_DAYS} days prior to each post-IPO fundraising event.")
            
            if not fund_checks.empty:
                f_checks_show = fund_checks.copy()
                if selected_entity != "All": f_checks_show = f_checks_show[f_checks_show["Name of REIT"] == selected_entity]
                
                # Show full table of evidence
                st.dataframe(f_checks_show, width="stretch", hide_index=True)
                
                # Show alerts if any failures
                failures = f_checks_show[f_checks_show["Status"].str.contains("Fail")]
                if not failures.empty:
                    st.error(f"Found {len(failures)} fundraising events without valid prior valuations.")
                else:
                    st.success("All fundraising events compliant.")
            else:
                st.info("No post-IPO fundraising data found to check.")

def render_valuation():
    render()