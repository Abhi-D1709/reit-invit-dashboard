# tabs/governance.py
from __future__ import annotations

import math
import re
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Defaults / wiring (prefers utils.common, falls back to hard-coded URL)
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
# Helpers: cleaners, classifiers, and Google Sheet readers
# --------------------------------------------------------------------
def _clean_str(x) -> str:
    s = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    s = re.sub(r"\s+", " ", s.replace("\u00A0", " ")).strip()
    return s


def _as_bool(x) -> bool:
    s = _clean_str(x).lower()
    return s in {"y", "yes", "true", "1"}


def _is_independent(type_cell: str) -> bool:
    """
    Independent iff the value STARTS with 'independent' (case/spacing/hyphens tolerant).
    If it starts with 'non independent', it's not independent.
    """
    s = _clean_str(type_cell).lower().replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    if s.startswith("non independent"):
        return False
    return s.startswith("independent")


def _is_non_exec(role_cell: str) -> bool:
    s = _clean_str(role_cell).lower().replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s.startswith("non executive")


def _is_director(type_cell: str) -> bool:
    """
    Count only rows whose 'Type of Members of Committee' explicitly contains 'director'.
    This excludes CEO/CXO/Company Secretary/etc.
    """
    return "director" in _clean_str(type_cell).lower()


def _filter_directors(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Type of Members of Committee"].apply(_is_director)].copy()


def _base_from_view_url(url: str) -> str:
    url = _clean_str(url)
    if "/edit" in url:
        return url.split("/edit")[0]
    if "/view" in url:
        return url.split("/view")[0]
    return url.rstrip("/")


@st.cache_data(show_spinner=False)
def read_google_sheet_by_sheetname(url: str, sheet_name: str) -> pd.DataFrame:
    """
    Reads a publicly shared Google Sheet (viewer link) using the 'gviz/tq' endpoint
    specifying the sheet name, which avoids needing gid.
    """
    import urllib.parse as _u

    base = _base_from_view_url(url)
    q = _u.quote(sheet_name, safe="")
    gviz = f"{base}/gviz/tq?tqx=out:csv&sheet={q}"
    df = pd.read_csv(gviz, dtype=str).applymap(_clean_str)
    return df


@st.cache_data(show_spinner=False)
def read_google_sheet_csv_default(url: str) -> pd.DataFrame:
    """
    Fallback: reads the default (first) sheet by CSV export.
    """
    base = _base_from_view_url(url)
    csv_url = f"{base}/export?format=csv"
    df = pd.read_csv(csv_url, dtype=str).applymap(_clean_str)
    return df


def _load_comp_meetings_board(url: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads:
      - Sheet1: composition
      - Sheet2: committee meetings
      - Sheet3: Board meetings (and possibly Independent Directors meeting date column)
    Returns (comp, meetings, board_meetings, warnings)
    """
    notes: List[str] = []
    # Sheet1
    try:
        comp = read_google_sheet_by_sheetname(url, "Sheet1")
    except Exception:
        comp = read_google_sheet_csv_default(url)
        notes.append("Sheet1 by name failed; used first sheet as composition.")

    # Sheet2
    meetings = pd.DataFrame()
    try:
        meetings = read_google_sheet_by_sheetname(url, "Sheet2")
    except Exception:
        notes.append("Sheet2 not found by name; committee meeting checks will be skipped.")

    # Sheet3
    board = pd.DataFrame()
    try:
        board = read_google_sheet_by_sheetname(url, "Sheet3")
    except Exception:
        notes.append("Sheet3 not found by name; Board/Independent Directors meeting checks will be skipped.")

    return comp, meetings, board, notes


# --------------------------------------------------------------------
# Core evaluation per-committee (DIRECTOR-ONLY counting where relevant)
# --------------------------------------------------------------------
def evaluate_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit Committee checks (composition):
      1. minimum 3 directors;
      2. at least 2/3 should be independent directors;
      3. at least 1 should have financial management expertise;
      4. Chairperson should be independent director.
    """
    if df.empty:
        return _empty_table("No rows for this committee/period.")

    df_dir = _filter_directors(df)
    members = len(df_dir)

    indep = df_dir["Type of Members of Committee"].apply(_is_independent).sum()
    indep_ratio_ok = indep * 3 >= members * 2 if members else False

    has_fin_exp = df_dir[
        "Is this member identified as having accounting or related Financial Management Expertise."
    ].apply(_as_bool).any()

    chair_rows_all = df[df["Is this Member the Chairperson for the Committee"].apply(_as_bool)]
    if chair_rows_all.empty:
        chair_indep = False
        chair_detail = "Chairperson: None found"
    else:
        chair_rows_dir = _filter_directors(chair_rows_all)
        if chair_rows_dir.empty:
            chair_indep = False
            chair_detail = "Chairperson is not a director"
        else:
            chair_indep = chair_rows_dir["Type of Members of Committee"].apply(_is_independent).all()
            chair_types = ", ".join(chair_rows_dir["Type of Members of Committee"].tolist())
            chair_detail = f"Chair type: {chair_types}"

    rows: List[Tuple[str, bool, str]] = [
        ("Min 3 directors", members >= 3, f"Members (directors only): {members}"),
        ("≥ 2/3 independent", bool(indep_ratio_ok),
         f"Independent (of directors): {indep}/{members} ({(indep/members*100 if members else 0):.0f}%)"),
        ("≥ 1 member has financial expertise", bool(has_fin_exp), "Yes" if has_fin_exp else "No"),
        ("Chairperson is independent", bool(chair_indep), chair_detail),
    ]
    return _to_table(rows)


def evaluate_nrc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nomination & Remuneration Committee (composition):
      1. Minimum 3 directors;
      2. All the directors – non-executive;
      3. At least 2/3 should be independent;
      4. Chairperson should be independent director.
    """
    if df.empty:
        return _empty_table("No rows for this committee/period.")

    df_dir = _filter_directors(df)
    members = len(df_dir)

    all_non_exec = df_dir["Role of Members of Committee"].apply(_is_non_exec).all() if members else False
    indep = df_dir["Type of Members of Committee"].apply(_is_independent).sum()
    indep_ratio_ok = indep * 3 >= members * 2 if members else False

    chair_rows_all = df[df["Is this Member the Chairperson for the Committee"].apply(_as_bool)]
    if chair_rows_all.empty:
        chair_indep = False
        chair_detail = "Chairperson: None found"
    else:
        chair_rows_dir = _filter_directors(chair_rows_all)
        if chair_rows_dir.empty:
            chair_indep = False
            chair_detail = "Chairperson is not a director"
        else:
            chair_indep = chair_rows_dir["Type of Members of Committee"].apply(_is_independent).all()
            chair_types = ", ".join(chair_rows_dir["Type of Members of Committee"].tolist())
            chair_detail = f"Chair type: {chair_types}"

    rows: List[Tuple[str, bool, str]] = [
        ("Min 3 directors", members >= 3, f"Members (directors only): {members}"),
        ("All non-executive", bool(all_non_exec),
         "All non-executive (directors)" if all_non_exec else "Found executive director(s)"),
        ("≥ 2/3 independent", bool(indep_ratio_ok),
         f"Independent (of directors): {indep}/{members} ({(indep/members*100 if members else 0):.0f}%)"),
        ("Chairperson is independent", bool(chair_indep), chair_detail),
    ]
    return _to_table(rows)


def evaluate_src(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stakeholders Relationship Committee (composition):
      1. Chairperson should be non-executive;
      2. Minimum 3 directors;
      3. Minimum 1 independent director.
    """
    if df.empty:
        return _empty_table("No rows for this committee/period.")

    df_dir = _filter_directors(df)
    members = len(df_dir)
    indep = df_dir["Type of Members of Committee"].apply(_is_independent).sum()

    chair_rows_all = df[df["Is this Member the Chairperson for the Committee"].apply(_as_bool)]
    if chair_rows_all.empty:
        chair_non_exec = False
        chair_detail = "Chairperson: None found"
    else:
        chair_rows_dir = _filter_directors(chair_rows_all)
        if chair_rows_dir.empty:
            chair_non_exec = False
            chair_detail = "Chairperson is not a director"
        else:
            chair_non_exec = chair_rows_dir["Role of Members of Committee"].apply(_is_non_exec).all()
            roles = ", ".join(chair_rows_dir["Role of Members of Committee"].tolist())
            chair_detail = f"Chair role: {roles}"

    rows: List[Tuple[str, bool, str]] = [
        ("Chairperson is non-executive", bool(chair_non_exec), chair_detail),
        ("Min 3 directors", members >= 3, f"Members (directors only): {members}"),
        ("≥ 1 independent", indep >= 1, f"Independent (of directors): {indep}/{members}"),
    ]
    return _to_table(rows)


def evaluate_rmc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Risk Management Committee (composition):
      1. Minimum 3 directors;
      2. Minimum 1 independent director.
    """
    if df.empty:
        return _empty_table("No rows for this committee/period.")

    df_dir = _filter_directors(df)
    members = len(df_dir)
    indep = df_dir["Type of Members of Committee"].apply(_is_independent).sum()

    rows: List[Tuple[str, bool, str]] = [
        ("Min 3 directors", members >= 3, f"Members (directors only): {members}"),
        ("≥ 1 independent", indep >= 1, f"Independent (of directors): {indep}/{members}"),
    ]
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
# Meeting rules / evaluation (Sheet2 — committees)
# --------------------------------------------------------------------
_MEETING_RULES: Dict[str, Dict[str, Optional[int]]] = {
    # per FY minimum meetings; gap thresholds in days (None = no rule)
    "Audit Committee": {"min_meetings": 4, "gap_days": 120, "min_indep_present": 2},
    "Nomination and Remuneration Committee": {"min_meetings": 1, "min_indep_present": 1},
    "Stakeholders Relationship Committee": {"min_meetings": 1, "min_indep_present": 1},
    "Risk Management Committee": {"min_meetings": 2, "gap_days": 210, "min_indep_present": 1},
}


def _committee_size_from_comp(comp_now: pd.DataFrame, committee: str) -> int:
    sub = comp_now[comp_now["Type of Committee"].str.lower() == committee.lower()]
    return len(_filter_directors(sub))


def _parse_meeting_dates(s: pd.Series) -> pd.Series:
    # Robust parsing for dd/mm/yyyy, d-m-y, etc.
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)


def evaluate_meetings_for_committee(
    comp_now: pd.DataFrame,
    meetings_fy: pd.DataFrame,
    committee: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Returns (summary_table, per_meeting_table, all_ok_flag)
    - Uses committee size from composition in the selected Period (comp_now).
    - Evaluates all meetings in the selected FY (meetings_fy).
    - Suppresses 'Max gap between meetings' row when the rule is not applicable.
    """
    rules = _MEETING_RULES[committee]
    size = _committee_size_from_comp(comp_now, committee)
    quorum_needed = max(2, math.ceil(size / 3)) if size else None

    # Subset meetings of this committee in the FY
    m = meetings_fy[meetings_fy["Type of Committee"].str.lower() == committee.lower()].copy()

    # --- If no meetings in FY, return an expectation-only summary
    if m.empty:
        summary_rows = [
            {"Rule": "Meetings in FY", "Expected": rules["min_meetings"], "Observed/Status": "0 (🔴)"},
            {
                "Rule": "Quorum per meeting",
                "Expected": quorum_needed if quorum_needed is not None else "n/a (no committee size)",
                "Observed/Status": "—",
            },
            {
                "Rule": "Min independent directors per meeting",
                "Expected": rules.get("min_indep_present", "—"),
                "Observed/Status": "—",
            },
        ]
        if rules.get("gap_days") is not None:
            summary_rows.append(
                {"Rule": "Max gap between meetings (days)", "Expected": rules["gap_days"], "Observed/Status": "—"}
            )
        out_summary = pd.DataFrame(summary_rows)
        return out_summary, pd.DataFrame(), False

    # --- Parse and sort meeting dates
    m["Meeting Date"] = _parse_meeting_dates(m["Date of Meeting of Committee"])
    m = m.sort_values("Meeting Date")

    # Frequency rule (FY)
    meet_cnt = len(m)
    freq_ok = meet_cnt >= int(rules["min_meetings"])

    # Gap rule (FY) – compute only if defined
    gap_days_rule = rules.get("gap_days")
    gap_ok = True
    worst_gap = None
    if gap_days_rule is not None and meet_cnt >= 2:
        diffs = (m["Meeting Date"].diff().dt.days).iloc[1:]
        if not diffs.empty:
            worst_gap = int(diffs.max())
            gap_ok = bool((diffs <= gap_days_rule).all())

    # Per-meeting quorum / independents
    per_rows = []
    all_meets_ok = True
    for _, row in m.iterrows():
        present = pd.to_numeric(row.get("Total No. of Members Present in the Meeting", ""), errors="coerce")
        indep_present = pd.to_numeric(row.get("Total No. of Independent directors in the meeting", ""), errors="coerce")
        present = int(present) if not pd.isna(present) else None
        indep_present = int(indep_present) if not pd.isna(indep_present) else None

        q_needed = quorum_needed if quorum_needed is not None else "-"
        q_ok = (present is not None and quorum_needed is not None and present >= quorum_needed)

        id_needed = _MEETING_RULES[committee].get("min_indep_present")
        id_ok = (indep_present is not None and id_needed is not None and indep_present >= id_needed)

        per_rows.append(
            {
                "Date": row["Meeting Date"].date() if pd.notna(row["Meeting Date"]) else "—",
                "Members Present": present if present is not None else "—",
                "Independent Present": indep_present if indep_present is not None else "—",
                "Quorum Needed": q_needed,
                "Quorum OK": "🟢" if q_ok else "🔴",
                "Independents Needed": id_needed if id_needed is not None else "—",
                "IDs OK": "🟢" if id_ok else "🔴",
            }
        )
        all_meets_ok = all_meets_ok and q_ok and id_ok

    per_table = pd.DataFrame(per_rows)

    # Build summary rows dynamically
    summary_rows = [
        {"Rule": "Meetings in FY", "Expected": _MEETING_RULES[committee]["min_meetings"],
         "Observed/Status": f"{meet_cnt} ({'🟢' if freq_ok else '🔴'})"},
        {
            "Rule": "Quorum per meeting",
            "Expected": quorum_needed if quorum_needed is not None else "n/a (no committee size)",
            "Observed/Status": "OK" if per_table["Quorum OK"].eq("🟢").all() else "🔴 Some meetings fail quorum",
        },
        {
            "Rule": "Min independent directors per meeting",
            "Expected": _MEETING_RULES[committee].get("min_indep_present", "—"),
            "Observed/Status": "OK" if per_table["IDs OK"].eq("🟢").all() else "🔴 Some meetings lack IDs",
        },
    ]
    if gap_days_rule is not None:
        gap_text = (f"Worst gap: {worst_gap} (OK≤{gap_days_rule})" if worst_gap is not None else "—")
        summary_rows.append(
            {"Rule": "Max gap between meetings (days)", "Expected": gap_days_rule,
             "Observed/Status": ("🟢 " + gap_text) if gap_ok else ("🔴 " + gap_text)}
        )

    summary = pd.DataFrame(summary_rows)

    # Overall flag: include gap rule only when applicable
    all_ok = freq_ok and all_meets_ok if gap_days_rule is None else (freq_ok and gap_ok and all_meets_ok)
    return summary, per_table, all_ok


# --------------------------------------------------------------------
# BOARD OF DIRECTORS & INDEPENDENT DIRECTORS MEETINGS (Sheet3)
# --------------------------------------------------------------------
def _board_size_from_sheet1_or_observed(comp_e: pd.DataFrame, board_fy: pd.DataFrame) -> Tuple[Optional[int], str]:
    """
    Board size from Sheet1 if 'Board of Directors' exists; else fallback to
    the max 'Total No. of Directors Present in the Meeting' observed in Sheet3.
    Returns (size, provenance).
    """
    # Try Sheet1 ("Board of Directors" as a 'Type of Committee', counted as directors only)
    comp_board = comp_e[comp_e["Type of Committee"].str.lower() == "board of directors"]
    size1 = len(_filter_directors(comp_board))
    if size1:
        return size1, "from Sheet1 ('Board of Directors')"

    # Fallback: observed maximum present in the FY
    if not board_fy.empty and "Total No. of Directors Present in the Meeting" in board_fy.columns:
        observed = pd.to_numeric(
            board_fy["Total No. of Directors Present in the Meeting"], errors="coerce"
        )
        if observed.notna().any():
            return int(observed.max()), "derived from max directors present in Sheet3"
    return None, "unavailable"


def _find_independent_meeting_date_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if "independent" in lc and "meeting" in lc and "date" in lc:
            cols.append(c)
    return cols


def evaluate_board_meetings(
    comp_e_fy: pd.DataFrame,
    board_fy: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Board checks:
      - >= 4 meetings in FY
      - gap <= 120 days
      - quorum per meeting: max(3, ceil(BoardSize/3))
      - >= 1 Independent Director present
    """
    if board_fy.empty:
        summary = pd.DataFrame(
            [
                {"Rule": "Board meetings in FY", "Expected": 4, "Observed/Status": "0 (🔴)"},
                {"Rule": "Quorum per meeting", "Expected": "max(3, ceil(BoardSize/3))", "Observed/Status": "—"},
                {"Rule": "Independent Director present (each mtg)", "Expected": "≥ 1", "Observed/Status": "—"},
                {"Rule": "Max gap between meetings (days)", "Expected": 120, "Observed/Status": "—"},
            ]
        )
        return summary, pd.DataFrame(), False

    # Parse dates and sort
    board = board_fy.copy()
    board["Meeting Date"] = _parse_meeting_dates(board["Date of Board Meeting"])
    board = board.sort_values("Meeting Date")

    # Frequency
    meet_cnt = len(board)
    freq_ok = meet_cnt >= 4

    # Gap rule (120)
    worst_gap = None
    gap_ok = True
    if meet_cnt >= 2:
        diffs = (board["Meeting Date"].diff().dt.days).iloc[1:]
        if not diffs.empty:
            worst_gap = int(diffs.max())
            gap_ok = bool((diffs <= 120).all())

    # Quorum & IDs per meeting
    board_size, prov = _board_size_from_sheet1_or_observed(comp_e_fy, board)
    quorum_needed = max(3, math.ceil(board_size / 3)) if board_size else None

    per_rows = []
    all_meets_ok = True
    for _, r in board.iterrows():
        present = pd.to_numeric(r.get("Total No. of Directors Present in the Meeting", ""), errors="coerce")
        indep_present = pd.to_numeric(r.get("Total No. of Independent directors in the meeting", ""), errors="coerce")
        present = int(present) if not pd.isna(present) else None
        indep_present = int(indep_present) if not pd.isna(indep_present) else None

        q_ok = (present is not None and quorum_needed is not None and present >= quorum_needed)
        id_ok = (indep_present is not None and indep_present >= 1)

        per_rows.append(
            {
                "Date": r["Meeting Date"].date() if pd.notna(r["Meeting Date"]) else "—",
                "Directors Present": present if present is not None else "—",
                "Independent Present": indep_present if indep_present is not None else "—",
                "Quorum Needed": quorum_needed if quorum_needed is not None else f"n/a ({prov})",
                "Quorum OK": "🟢" if q_ok else "🔴",
                "≥1 ID Present": "🟢" if id_ok else "🔴",
            }
        )
        all_meets_ok = all_meets_ok and q_ok and id_ok

    per_table = pd.DataFrame(per_rows)

    gap_text = (f"Worst gap: {worst_gap} (OK≤120)" if worst_gap is not None else "—")
    summary = pd.DataFrame(
        [
            {"Rule": "Board meetings in FY", "Expected": 4, "Observed/Status": f"{meet_cnt} ({'🟢' if freq_ok else '🔴'})"},
            {
                "Rule": "Quorum per meeting",
                "Expected": f"max(3, ceil(BoardSize/3)) [{prov}]",
                "Observed/Status": "OK" if per_table["Quorum OK"].eq("🟢").all() else "🔴 Some meetings fail quorum",
            },
            {
                "Rule": "Independent Director present (each mtg)",
                "Expected": "≥ 1",
                "Observed/Status": "OK" if per_table["≥1 ID Present"].eq("🟢").all() else "🔴 Some meetings lack IDs",
            },
            {"Rule": "Max gap between meetings (days)", "Expected": 120,
             "Observed/Status": ("🟢 " + gap_text) if gap_ok else ("🔴 " + gap_text)},
        ]
    )
    all_ok = freq_ok and gap_ok and all_meets_ok
    return summary, per_table, all_ok


def evaluate_independent_directors_meeting(board_fy: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Finds any Sheet3 column like '*Independent*Meeting*Date*' (case-insensitive) and checks
    at least 1 meeting date present in the FY.
    """
    if board_fy.empty:
        return pd.DataFrame([{"Rule": "Independent Directors’ meeting in FY", "Expected": 1, "Observed/Status": "0 (🔴)"}]), False

    date_cols = _find_independent_meeting_date_cols(board_fy)
    if not date_cols:
        return pd.DataFrame([{
            "Rule": "Independent Directors’ meeting in FY",
            "Expected": 1,
            "Observed/Status": "— (column not found in Sheet3)"
        }]), False

    # Count non-null dates across any detected column
    found = 0
    for c in date_cols:
        d = pd.to_datetime(board_fy[c], errors="coerce", dayfirst=True, infer_datetime_format=True)
        found += d.notna().sum()

    ok = found >= 1
    return pd.DataFrame([{
        "Rule": "Independent Directors’ meeting in FY",
        "Expected": 1,
        "Observed/Status": f"{found} ({'🟢' if ok else '🔴'})"
    }]), ok


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
def render() -> None:
    st.title("Governance — Committee Composition Checks")

    seg = st.sidebar.selectbox("Select Segment", ["REIT"], index=0)
    _ = seg  # reserved for future InvIT support

    url = st.sidebar.text_input(
        "Data URL (Google Sheet - public view)", value=DEFAULT_GOVERNANCE_URL
    )

    comp, meetings, board, notes = _load_comp_meetings_board(url)
    if notes:
        for n in notes:
            st.info(n)

    if comp.empty:
        st.warning("No rows loaded from the composition sheet.")
        return

    # Validate required columns (composition)
    required_comp = [
        "Name of REIT",
        "Financial Year",
        "Period Ended",
        "Type of Committee",
        "Name of Member of Committee",
        "Type of Members of Committee",
        "Role of Members of Committee",
        "Is this member identified as having accounting or related Financial Management Expertise.",
        "Is this Member the Chairperson for the Committee",
    ]
    missing = [c for c in required_comp if c not in comp.columns]
    if missing:
        st.error(f"Missing columns in Sheet1 (composition): {missing}")
        st.dataframe(comp.head())
        return

    comp = comp.applymap(_clean_str)

    # Optional meetings sheet validations (Sheet2)
    meetings_ok = False
    if not meetings.empty:
        required_meet = [
            "Name of REIT",
            "Financial Year",
            "Period Ended",
            "Type of Committee",
            "Date of Meeting of Committee",
            "Total No. of Members Present in the Meeting",
            "Total No. of Independent directors in the meeting",
        ]
        m_missing = [c for c in required_meet if c not in meetings.columns]
        if m_missing:
            st.warning(f"Meetings sheet found but missing columns: {m_missing}. Meeting checks disabled.")
        else:
            meetings = meetings.applymap(_clean_str)
            meetings_ok = True

    # Optional board sheet validations (Sheet3)
    board_ok = False
    board_cols_needed = [
        "Name of REIT",
        "Financial Year",
        "Period Ended",
        "Date of Board Meeting",
        "Total No. of Directors Present in the Meeting",
        "Total No. of Independent directors in the meeting",
    ]
    if not board.empty:
        miss3 = [c for c in board_cols_needed if c not in board.columns]
        if miss3:
            st.warning(f"Sheet3 found but missing key columns: {miss3}. Board/ID checks may be partial.")
        board = board.applymap(_clean_str)
        board_ok = True

    # Selections
    entity = st.selectbox("Choose REIT", sorted(comp["Name of REIT"].unique()))
    comp_e = comp[comp["Name of REIT"] == entity]

    years = sorted(comp_e["Financial Year"].unique())
    fy = st.selectbox("Financial Year", years, index=max(0, len(years) - 1))
    comp_ey = comp_e[comp_e["Financial Year"] == fy]

    periods = sorted(comp_ey["Period Ended"].unique())
    period = st.selectbox("Period Ended", periods, index=0)
    comp_now = comp_ey[comp_ey["Period Ended"] == period]

    # Meetings subset for this entity & FY (Sheet2 — not filtered by period; frequency/gap are FY-level)
    meetings_fy = pd.DataFrame()
    if meetings_ok:
        meetings_fy = meetings[(meetings["Name of REIT"] == entity) & (meetings["Financial Year"] == fy)].copy()

    # Board subset for this entity & FY (Sheet3)
    board_fy = pd.DataFrame()
    if board_ok:
        board_fy = board[(board["Name of REIT"] == entity) & (board["Financial Year"] == fy)].copy()

    # ----- Committee composition + meeting checks (Sheet2)
    committees = {
        "Audit Committee": evaluate_audit,
        "Nomination and Remuneration Committee": evaluate_nrc,
        "Stakeholders Relationship Committee": evaluate_src,
        "Risk Management Committee": evaluate_rmc,
    }

    for title, comp_fn in committees.items():
        st.subheader(title)

        comp_sub = comp_now[comp_now["Type of Committee"].str.lower() == title.lower()]
        st.table(comp_fn(comp_sub))

        # Meeting checks
        if meetings_fy.empty:
            st.info("Meeting sheet not available or no rows for this REIT/FY.")
            continue

        st.markdown(f"**Meetings — {title} (FY {fy})**")
        summary, per_meeting, ok = evaluate_meetings_for_committee(comp_now, meetings_fy, title)
        st.table(summary)
        if not per_meeting.empty:
            st.dataframe(per_meeting, use_container_width=True)
        if ok:
            st.success("All meeting rules satisfied for the FY (given the selected period's committee size).")
        else:
            st.error("One or more meeting rules are not satisfied for the FY.")

    # ----- Board of Directors & Independent Directors’ meeting (Sheet3)
    st.subheader("Board of Directors — Meetings (FY level)")
    if board_fy.empty:
        st.info("Sheet3 not available or no Board rows for this REIT/FY.")
    else:
        summary_b, per_b, ok_b = evaluate_board_meetings(comp_ey, board_fy)
        st.table(summary_b)
        if not per_b.empty:
            st.dataframe(per_b, use_container_width=True)
        if ok_b:
            st.success("All Board meeting rules satisfied for the FY.")
        else:
            st.error("One or more Board meeting rules are not satisfied for the FY.")

        # Independent Directors’ meeting
        st.markdown("**Independent Directors’ Meeting (FY level)**")
        id_summary, id_ok = evaluate_independent_directors_meeting(board_fy)
        st.table(id_summary)
        if id_ok:
            st.success("Independent Directors met at least once in the FY.")
        else:
            st.error("Independent Directors did NOT meet at least once in the FY, or the date column is missing.")

def render_governance() -> None:
    render()
