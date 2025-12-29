# tabs/governance.py
from __future__ import annotations

import re
import math
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Defaults / wiring
# --------------------------------------------------------------------
_HARDCODED = (
    "https://docs.google.com/spreadsheets/d/1ETx5UZKQQyZKxkF4fFJ4R9wa7i7TNp7EXIhHWiVYG7s/edit?usp=sharing"
)
DEFAULT_GOVERNANCE_URL = _HARDCODED
try:
    from utils import common as _common  # type: ignore

    _cfg = getattr(_common, "GOVERNANCE_REIT_SHEET_URL", "").strip()
    if _cfg:
        DEFAULT_GOVERNANCE_URL = _cfg
except Exception:
    pass


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _clean_str(x) -> str:
    s = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    s = re.sub(r"\s+", " ", s.replace("\u00A0", " ")).strip()
    return s

def _as_bool(x) -> bool:
    s = _clean_str(x).lower()
    return s in {"y", "yes", "true", "1"}

def _is_independent(type_cell: str) -> bool:
    s = _clean_str(type_cell).lower().replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    if s.startswith("non independent"):
        return False
    return s.startswith("independent")

def _is_non_exec(role_cell: str) -> bool:
    s = _clean_str(role_cell).lower().replace("-", " ")
    return s.startswith("non executive")

def _is_director(type_cell: str) -> bool:
    s = _clean_str(type_cell).lower()
    return "director" in s

def _filter_directors(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Type of Members of Committee"].apply(_is_director)].copy()

def _parse_date(x):
    return pd.to_datetime(x, dayfirst=True, errors='coerce')

@st.cache_data(show_spinner=False)
def read_google_sheet_excel(url: str, sheet_name: str) -> pd.DataFrame:
    """
    Reads a specific sheet from a Google Sheet URL by exporting as XLSX.
    Requires 'openpyxl' installed.
    """
    url = _clean_str(url)
    if "/edit" in url:
        base = url.split("/edit")[0]
    elif "/view" in url:
        base = url.split("/view")[0]
    else:
        base = url.rstrip("/")
    
    # Export as XLSX to support multiple sheets by name
    xlsx_url = f"{base}/export?format=xlsx"
    
    try:
        df = pd.read_excel(xlsx_url, sheet_name=sheet_name, dtype=str)
        return df.applymap(_clean_str)
    except Exception as e:
        st.error(f"Error reading sheet '{sheet_name}'. Ensure the sheet exists and the URL is public. Details: {e}")
        return pd.DataFrame()


# --------------------------------------------------------------------
# Composition Evaluation (Sheet 1)
# --------------------------------------------------------------------
def evaluate_composition_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return _empty_table("No composition data.")
    df_dir = _filter_directors(df)
    members = len(df_dir)
    indep = df_dir["Type of Members of Committee"].apply(_is_independent).sum()
    indep_ratio_ok = indep * 3 >= members * 2 if members else False
    has_fin_exp = df_dir["Is this member identified as having accounting or related Financial Management Expertise."].apply(_as_bool).any()
    
    chair_rows = df[df["Is this Member the Chairperson for the Committee"].apply(_as_bool)]
    if chair_rows.empty:
        chair_check = False
        chair_detail = "None found"
    else:
        # Check if Chair is Independent Director
        c_r = chair_rows.iloc[0] # assume 1 chair
        is_dir = _is_director(c_r["Type of Members of Committee"])
        is_ind = _is_independent(c_r["Type of Members of Committee"])
        chair_check = is_dir and is_ind
        chair_detail = c_r["Type of Members of Committee"]

    rows = [
        ("Min 3 directors", members >= 3, f"Directors: {members}"),
        ("≥ 2/3 independent", bool(indep_ratio_ok), f"Indep: {indep}/{members}"),
        ("≥ 1 member has fin. expertise", bool(has_fin_exp), "Yes" if has_fin_exp else "No"),
        ("Chairperson is Indep. Director", bool(chair_check), chair_detail),
    ]
    return _to_table(rows)

def evaluate_composition_nrc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return _empty_table("No composition data.")
    df_dir = _filter_directors(df)
    members = len(df_dir)
    all_non_exec = df_dir["Role of Members of Committee"].apply(_is_non_exec).all() if members else False
    indep = df_dir["Type of Members of Committee"].apply(_is_independent).sum()
    indep_ratio_ok = indep * 3 >= members * 2 if members else False
    
    chair_rows = df[df["Is this Member the Chairperson for the Committee"].apply(_as_bool)]
    if chair_rows.empty:
        chair_check = False
        chair_detail = "None found"
    else:
        c_r = chair_rows.iloc[0]
        chair_check = _is_director(c_r["Type of Members of Committee"]) and _is_independent(c_r["Type of Members of Committee"])
        chair_detail = c_r["Type of Members of Committee"]

    rows = [
        ("Min 3 directors", members >= 3, f"Directors: {members}"),
        ("All directors non-executive", bool(all_non_exec), "Yes" if all_non_exec else "No"),
        ("≥ 2/3 independent", bool(indep_ratio_ok), f"Indep: {indep}/{members}"),
        ("Chairperson is Indep. Director", bool(chair_check), chair_detail),
    ]
    return _to_table(rows)

def evaluate_composition_src(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return _empty_table("No composition data.")
    df_dir = _filter_directors(df)
    members = len(df_dir)
    indep = df_dir["Type of Members of Committee"].apply(_is_independent).sum()
    
    chair_rows = df[df["Is this Member the Chairperson for the Committee"].apply(_as_bool)]
    if chair_rows.empty:
        chair_check = False
        chair_detail = "None found"
    else:
        c_r = chair_rows.iloc[0]
        chair_check = _is_non_exec(c_r["Role of Members of Committee"])
        chair_detail = c_r["Role of Members of Committee"]

    rows = [
        ("Chairperson is non-executive", bool(chair_check), chair_detail),
        ("Min 3 directors", members >= 3, f"Directors: {members}"),
        ("≥ 1 independent", indep >= 1, f"Indep: {indep}"),
    ]
    return _to_table(rows)

def evaluate_composition_rmc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return _empty_table("No composition data.")
    df_dir = _filter_directors(df)
    members = len(df_dir)
    indep = df_dir["Type of Members of Committee"].apply(_is_independent).sum()

    rows = [
        ("Min 3 directors", members >= 3, f"Directors: {members}"),
        ("≥ 1 independent", indep >= 1, f"Indep: {indep}"),
    ]
    return _to_table(rows)


# --------------------------------------------------------------------
# Meeting Evaluation (Sheet 2)
# --------------------------------------------------------------------
def evaluate_meetings(committee_type: str, df_meet: pd.DataFrame, composition_count: int) -> pd.DataFrame:
    """
    Generic logic to check meeting compliance.
    """
    if df_meet.empty:
        return _empty_table("No meeting data found for this FY.")

    # Convert dates
    df_meet = df_meet.copy()
    df_meet["Date"] = _parse_date(df_meet["Date of Meeting of Committee"])
    df_meet = df_meet.sort_values("Date")
    
    # Parse numbers
    df_meet["Total Present"] = pd.to_numeric(df_meet["Total No. of Members Present in the Meeting"], errors='coerce').fillna(0)
    df_meet["IDs Present"] = pd.to_numeric(df_meet["Total No. of Independent directors in the meeting"], errors='coerce').fillna(0)

    dates = df_meet["Date"].dropna().tolist()
    num_meetings = len(dates)
    
    # -- Rules Configuration --
    min_meetings = 0
    max_gap_days = None
    min_ids_present = 0
    quorum_check = False
    
    if "Audit" in committee_type:
        min_meetings = 4
        max_gap_days = 120
        min_ids_present = 2
        quorum_check = True
    elif "Nomination" in committee_type:
        min_meetings = 1
        min_ids_present = 1
        quorum_check = True
    elif "Stakeholders" in committee_type:
        min_meetings = 1
        min_ids_present = 1
        quorum_check = False # Prompt didn't strictly specify quorum for SRC, just "At least 1 ID present"
    elif "Risk" in committee_type:
        min_meetings = 2
        max_gap_days = 210
        min_ids_present = 1
        quorum_check = True

    rows = []

    # 1. Frequency Check
    rows.append((f"Meet at least {min_meetings} times in FY", num_meetings >= min_meetings, f"Count: {num_meetings}"))

    # 2. Gap Check
    if max_gap_days and num_meetings > 1:
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        max_actual_gap = max(gaps) if gaps else 0
        rows.append((f"Gap ≤ {max_gap_days} days", max_actual_gap <= max_gap_days, f"Max Gap: {max_actual_gap} days"))
    elif max_gap_days:
        # Less than 2 meetings means gap check is N/A or technically pass if freq failed
        rows.append((f"Gap ≤ {max_gap_days} days", True, "N/A (< 2 meetings)"))

    # 3. Quorum Check (Min 2 or 1/3rd)
    if quorum_check:
        # Quorum = max(2, ceil(Total/3))
        required_quorum = max(2, math.ceil(composition_count / 3))
        failed_quorum = df_meet[df_meet["Total Present"] < required_quorum]
        passed = failed_quorum.empty
        detail = f"Req: {required_quorum} (Total Composition: {composition_count})"
        if not passed:
            bad_dates = [d.strftime('%Y-%m-%d') for d in failed_quorum["Date"]]
            detail += f" | Failed on: {', '.join(bad_dates)}"
        rows.append(("Quorum (Min 2 or 1/3rd)", passed, detail))

    # 4. ID Presence Check
    if min_ids_present > 0:
        failed_ids = df_meet[df_meet["IDs Present"] < min_ids_present]
        passed = failed_ids.empty
        detail = f"Req IDs: {min_ids_present}"
        if not passed:
            bad_dates = [d.strftime('%Y-%m-%d') for d in failed_ids["Date"]]
            detail += f" | Failed on: {', '.join(bad_dates)}"
        rows.append((f"At least {min_ids_present} IDs present", passed, detail))

    return _to_table(rows)


def _to_table(rows: List[Tuple[str, bool, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Check": [r[0] for r in rows],
            "Result": ["🟢" if r[1] else "🔴" for r in rows],
            "Detail": [r[2] for r in rows],
        }
    )

def _empty_table(message: str) -> pd.DataFrame:
    return pd.DataFrame({"Check": [message], "Result": ["—"], "Detail": [""]})


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
def render() -> None:
    st.title("Governance — Committee Checks")

    st.info("Ensure the Google Sheet is visible to anyone with the link.")
    url = st.sidebar.text_input(
        "Data URL (Google Sheet)", value=DEFAULT_GOVERNANCE_URL
    )

    # Load Data (Sheet 1 and Sheet 2)
    # Using 'Sheet1' and 'Sheet2' explicitly as per prompt description
    df_comp_raw = read_google_sheet_excel(url, "Sheet1")
    df_meet_raw = read_google_sheet_excel(url, "Sheet2")

    if df_comp_raw.empty:
        st.warning("Could not load 'Sheet1' (Composition).")
        return

    # Basic Validation
    req_comp = ["Name of REIT", "Financial Year", "Period Ended", "Type of Committee"]
    if not all(c in df_comp_raw.columns for c in req_comp):
        st.error(f"Sheet1 Missing columns: {[c for c in req_comp if c not in df_comp_raw.columns]}")
        return

    # Filters
    entity = st.selectbox("Choose REIT", sorted(df_comp_raw["Name of REIT"].unique()))
    
    # Filter Comp by Entity
    df_c = df_comp_raw[df_comp_raw["Name of REIT"] == entity]
    
    years = sorted(df_c["Financial Year"].unique())
    fy = st.selectbox("Financial Year", years, index=max(0, len(years) - 1))
    
    # Filter Comp by FY
    df_cy = df_c[df_c["Financial Year"] == fy]
    
    periods = sorted(df_cy["Period Ended"].unique())
    period = st.selectbox("Period Ended (for Composition Snapshot)", periods, index=0)
    
    # Final Composition Snapshot
    df_comp_now = df_cy[df_cy["Period Ended"] == period]

    # Filter Meetings by Entity and FY (Meetings are checked for the whole FY)
    if not df_meet_raw.empty:
        df_m = df_meet_raw[df_meet_raw["Name of REIT"] == entity]
        df_my = df_m[df_m["Financial Year"] == fy]
    else:
        df_my = pd.DataFrame()

    # Define Processors
    committees = {
        "Audit Committee": (evaluate_composition_audit, "Audit Committee"),
        "Nomination and Remuneration Committee": (evaluate_composition_nrc, "Nomination and Remuneration Committee"),
        "Stakeholders Relationship Committee": (evaluate_composition_src, "Stakeholders Relationship Committee"),
        "Risk Management Committee": (evaluate_composition_rmc, "Risk Management Committee"),
    }

    for label, (comp_fn, filter_key) in committees.items():
        st.markdown(f"### {label}")
        
        # 1. Composition
        sub_comp = df_comp_now[df_comp_now["Type of Committee"].str.lower() == filter_key.lower()]
        
        # Calculate Total Members for Quorum usage
        # Note: We take the total member count from the composition snapshot
        # This includes non-directors if they are members, as quorum usually applies to 'members'
        total_members_count = len(sub_comp)

        c1, c2 = st.columns(2)
        
        with c1:
            st.caption("Composition Checks")
            st.table(comp_fn(sub_comp))

        # 2. Meetings
        with c2:
            st.caption("Meeting Compliance (FY)")
            if df_my.empty:
                st.info("No meeting data loaded (Sheet2).")
            else:
                sub_meet = df_my[df_my["Type of Committee"].str.lower() == filter_key.lower()]
                st.table(evaluate_meetings(label, sub_meet, total_members_count))
        
        st.divider()

def render_governance() -> None:
    render()