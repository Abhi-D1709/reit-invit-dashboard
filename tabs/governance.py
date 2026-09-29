# tabs/governance.py
from __future__ import annotations

import math
import re
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils import status
from utils.status import Status  # noqa: F401
from utils import rules as thresholds  # every threshold lives in utils/rules.py (aliased: `rules` is a local name below)

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
    # Only rows explicitly marked with '*director*' count as directors
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
    import urllib.parse as _u
    base = _base_from_view_url(url)
    q = _u.quote(sheet_name, safe="")
    csv_url = f"{base}/gviz/tq?tqx=out:csv&sheet={q}"
    df = pd.read_csv(csv_url, dtype=str).map(_clean_str)
    return df


@st.cache_data(show_spinner=False)
def read_google_sheet_csv_default(url: str) -> pd.DataFrame:
    base = _base_from_view_url(url)
    csv_url = f"{base}/export?format=csv"
    df = pd.read_csv(csv_url, dtype=str).map(_clean_str)
    return df


def _load_all_sheets(url: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Returns (composition Sheet1, committee meetings Sheet2, board meetings Sheet3,
             independent directors meetings Sheet4, notes)
    """
    notes: List[str] = []

    # Sheet1
    try:
        comp = read_google_sheet_by_sheetname(url, "Sheet1")
    except Exception:
        comp = read_google_sheet_csv_default(url)
        notes.append("Sheet1 by name failed; used first sheet as composition.")

    # Sheet2
    try:
        meetings = read_google_sheet_by_sheetname(url, "Sheet2")
    except Exception:
        meetings = pd.DataFrame()
        notes.append("Sheet2 not found; committee meeting checks will be skipped.")

    # Sheet3
    try:
        board = read_google_sheet_by_sheetname(url, "Sheet3")
    except Exception:
        board = pd.DataFrame()
        notes.append("Sheet3 not found; Board checks will be skipped.")

    # Sheet4 (Independent Directors)
    try:
        ind = read_google_sheet_by_sheetname(url, "Sheet4")
    except Exception:
        ind = pd.DataFrame()
        notes.append("Sheet4 not found; Independent Directors meeting check will be skipped.")

    return comp, meetings, board, ind, notes


# --------------------------------------------------------------------
# Core evaluation per-committee (DIRECTOR-ONLY counting where relevant)
# --------------------------------------------------------------------
_MEMBER_TYPE_COL = "Type of Members of Committee"
_MEMBER_ROLE_COL = "Role of Members of Committee"
_CHAIR_COL = "Is this Member the Chairperson for the Committee"


def _director_counts(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """(director rows, number of directors, number of independent directors)."""
    df_dir = _filter_directors(df)
    indep = df_dir[_MEMBER_TYPE_COL].apply(_is_independent).sum()
    return df_dir, len(df_dir), indep


def _chair_check(df: pd.DataFrame, column: str, predicate, label: str) -> Tuple[bool, str]:
    """(chairperson rows that are directors all satisfy `predicate` on `column`, detail text).
    Fails when no chairperson is marked or the marked chairperson is not a director."""
    chair_rows_all = df[df[_CHAIR_COL].apply(_as_bool)]
    if chair_rows_all.empty:
        return False, "Chairperson: None found"
    chair_rows_dir = _filter_directors(chair_rows_all)
    if chair_rows_dir.empty:
        return False, "Chairperson is not a director"
    chair_ok = chair_rows_dir[column].apply(predicate).all()
    return chair_ok, f"{label}: {', '.join(chair_rows_dir[column].tolist())}"


def _min_directors_row(members: int) -> tuple:
    return (f"Min {thresholds.COMMITTEE_MIN_DIRECTORS} directors", members >= thresholds.COMMITTEE_MIN_DIRECTORS,
            f"Members (directors only): {members}")


def _independent_share_row(indep: int, members: int) -> tuple:
    num, den = thresholds.COMMITTEE_INDEPENDENT_SHARE
    share_ok = indep * den >= members * num if members else False
    return (f"≥ {num}/{den} independent", bool(share_ok),
            f"Independent (of directors): {indep}/{members} ({(indep/members*100 if members else 0):.0f}%)")


def _min_independent_row(indep: int, members: int) -> tuple:
    return (f"≥ {thresholds.COMMITTEE_MIN_INDEPENDENT} independent", indep >= thresholds.COMMITTEE_MIN_INDEPENDENT,
            f"Independent (of directors): {indep}/{members}")


def evaluate_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_table("No rows for this committee/period.")

    df_dir, members, indep = _director_counts(df)
    has_fin_exp = df_dir[
        "Is this member identified as having accounting or related Financial Management Expertise."
    ].apply(_as_bool).any()
    chair_indep, chair_detail = _chair_check(df, _MEMBER_TYPE_COL, _is_independent, "Chair type")

    rows = [
        _min_directors_row(members),
        _independent_share_row(indep, members),
        ("≥ 1 member has financial expertise", bool(has_fin_exp), "Yes" if has_fin_exp else "No"),
        ("Chairperson is independent", bool(chair_indep), chair_detail),
    ]
    return _to_table(rows)


def evaluate_nrc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_table("No rows for this committee/period.")

    df_dir, members, indep = _director_counts(df)
    all_non_exec = df_dir[_MEMBER_ROLE_COL].apply(_is_non_exec).all() if members else False
    chair_indep, chair_detail = _chair_check(df, _MEMBER_TYPE_COL, _is_independent, "Chair type")

    rows = [
        _min_directors_row(members),
        ("All non-executive", bool(all_non_exec),
         "All non-executive (directors)" if all_non_exec else "Found executive director(s)"),
        _independent_share_row(indep, members),
        ("Chairperson is independent", bool(chair_indep), chair_detail),
    ]
    return _to_table(rows)


def evaluate_src(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_table("No rows for this committee/period.")

    _, members, indep = _director_counts(df)
    chair_non_exec, chair_detail = _chair_check(df, _MEMBER_ROLE_COL, _is_non_exec, "Chair role")

    rows = [
        ("Chairperson is non-executive", bool(chair_non_exec), chair_detail),
        _min_directors_row(members),
        _min_independent_row(indep, members),
    ]
    return _to_table(rows)


def evaluate_rmc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_table("No rows for this committee/period.")

    _, members, indep = _director_counts(df)

    rows = [
        _min_directors_row(members),
        _min_independent_row(indep, members),
    ]
    return _to_table(rows)


def _to_table(rows: List[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Check": [r[0] for r in rows], "Result": [status.PASS_TAG if r[1] else status.FAIL_TAG for r in rows], "Detail": [r[2] for r in rows]}
    )


def _empty_table(message: str) -> pd.DataFrame:
    return pd.DataFrame({"Check": [message], "Result": ["—"], "Detail": [""]})


# --------------------------------------------------------------------
# Meeting rules / evaluation (Sheet2 — committees)
# --------------------------------------------------------------------
def _committee_size_from_comp(comp_now: pd.DataFrame, committee: str) -> int:
    sub = comp_now[comp_now["Type of Committee"].str.lower() == committee.lower()]
    return len(_filter_directors(sub))


def _parse_meeting_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


# --------------------------------------------------------------------
# Shared meeting algorithm (committees on Sheet2, Board on Sheet3):
# meeting count, max gap between meetings, and per-meeting quorum / independents present
# --------------------------------------------------------------------
_INDEP_PRESENT_COL = "Total No. of Independent directors in the meeting"


def _sorted_meetings(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Copy of `df` with a parsed "Meeting Date" column, sorted by it."""
    out = df.copy()
    out["Meeting Date"] = _parse_meeting_dates(out[date_col])
    return out.sort_values("Meeting Date")


def _max_gap_check(meeting_dates: pd.Series, max_gap_days) -> Tuple[bool, Optional[int]]:
    """(every gap between consecutive meetings <= max_gap_days, worst gap in days or None)."""
    gap_ok = True
    worst_gap = None
    if len(meeting_dates) >= 2:
        diffs = (meeting_dates.diff().dt.days).iloc[1:]
        if not diffs.empty:
            worst_gap = int(diffs.max())
            gap_ok = bool((diffs <= max_gap_days).all())
    return gap_ok, worst_gap


def _count_or_none(value) -> Optional[int]:
    num = pd.to_numeric(value, errors="coerce")
    return int(num) if not pd.isna(num) else None


def _meets_minimum(observed: Optional[int], needed: Optional[int]) -> bool:
    return observed is not None and needed is not None and observed >= needed


def _check_each_meeting(
    meetings: pd.DataFrame, present_col: str, quorum_needed: Optional[int], indep_needed: Optional[int]
) -> Tuple[List[Dict], bool]:
    """Per-meeting attendance checks. Returns one dict per meeting (date, present, indep_present,
    q_ok, id_ok) and whether every meeting met both quorum and the independents minimum."""
    results = []
    all_meets_ok = True
    for _, row in meetings.iterrows():
        present = _count_or_none(row.get(present_col, ""))
        indep_present = _count_or_none(row.get(_INDEP_PRESENT_COL, ""))
        q_ok = _meets_minimum(present, quorum_needed)
        id_ok = _meets_minimum(indep_present, indep_needed)
        results.append({
            "date": row["Meeting Date"].date() if pd.notna(row["Meeting Date"]) else "—",
            "present": present if present is not None else "—",
            "indep_present": indep_present if indep_present is not None else "—",
            "q_ok": q_ok,
            "id_ok": id_ok,
        })
        all_meets_ok = all_meets_ok and q_ok and id_ok
    return results, all_meets_ok


def _pass_fail(ok: bool) -> str:
    return status.PASS_TAG if ok else status.FAIL_TAG


def _all_pass_or(per_table: pd.DataFrame, column: str, failure_text: str) -> str:
    return "OK" if per_table[column].eq(status.PASS_TAG).all() else status.tag(status.Status.FAIL, failure_text)


def _gap_status(gap_ok: bool, worst_gap: Optional[int], max_gap_days) -> str:
    gap_text = (f"Worst gap: {worst_gap} (OK≤{max_gap_days})" if worst_gap is not None else "—")
    return status.tag(status.Status.PASS if gap_ok else status.Status.FAIL, gap_text)


def evaluate_meetings_for_committee(
    comp_now: pd.DataFrame,
    meetings_fy: pd.DataFrame,
    committee: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    rules = thresholds.GOVERNANCE_MEETING_RULES[committee]
    size = _committee_size_from_comp(comp_now, committee)
    quorum_needed = max(2, math.ceil(size / 3)) if size else None
    gap_days_rule = rules.get("gap_days")

    m = meetings_fy[meetings_fy["Type of Committee"].str.lower() == committee.lower()]

    if m.empty:
        summary_rows = [
            {"Rule": "Meetings in FY", "Expected": rules["min_meetings"], "Observed/Status": f"0 ({status.FAIL_TAG})"},
            {"Rule": "Quorum per meeting", "Expected": quorum_needed if quorum_needed is not None else "n/a", "Observed/Status": "—"},
            {"Rule": "Min independent directors per meeting", "Expected": rules.get("min_indep_present", "—"), "Observed/Status": "—"},
        ]
        if gap_days_rule is not None:
            summary_rows.append({"Rule": "Max gap between meetings (days)", "Expected": gap_days_rule, "Observed/Status": "—"})
        return pd.DataFrame(summary_rows), pd.DataFrame(), False

    m = _sorted_meetings(m, "Date of Meeting of Committee")

    meet_cnt = len(m)
    freq_ok = meet_cnt >= int(rules["min_meetings"])

    gap_ok, worst_gap = (True, None) if gap_days_rule is None else _max_gap_check(m["Meeting Date"], gap_days_rule)

    id_needed = rules.get("min_indep_present")
    checks, all_meets_ok = _check_each_meeting(
        m, "Total No. of Members Present in the Meeting", quorum_needed, id_needed
    )
    per_table = pd.DataFrame([
        {
            "Date": c["date"],
            "Members Present": c["present"],
            "Independent Present": c["indep_present"],
            "Quorum Needed": quorum_needed if quorum_needed is not None else "-",
            "Quorum OK": _pass_fail(c["q_ok"]),
            "Independents Needed": id_needed if id_needed is not None else "—",
            "IDs OK": _pass_fail(c["id_ok"]),
        }
        for c in checks
    ])

    summary_rows = [
        {"Rule": "Meetings in FY", "Expected": rules["min_meetings"],
         "Observed/Status": f"{meet_cnt} ({_pass_fail(freq_ok)})"},
        {"Rule": "Quorum per meeting", "Expected": quorum_needed if quorum_needed is not None else "n/a",
         "Observed/Status": _all_pass_or(per_table, "Quorum OK", "Some meetings fail quorum")},
        {"Rule": "Min independent directors per meeting", "Expected": rules.get("min_indep_present", "—"),
         "Observed/Status": _all_pass_or(per_table, "IDs OK", "Some meetings lack IDs")},
    ]
    if gap_days_rule is not None:
        summary_rows.append({"Rule": "Max gap between meetings (days)", "Expected": gap_days_rule,
                             "Observed/Status": _gap_status(gap_ok, worst_gap, gap_days_rule)})

    # gap_ok is True when the committee has no gap rule, so this covers both cases
    all_ok = freq_ok and gap_ok and all_meets_ok
    return pd.DataFrame(summary_rows), per_table, all_ok


# --------------------------------------------------------------------
# BOARD OF DIRECTORS (Sheet3) & INDEPENDENT DIRECTORS MEETING (Sheet4)
# --------------------------------------------------------------------
def _board_size_from_sheet1_or_observed(comp_e: pd.DataFrame, board_fy: pd.DataFrame) -> Tuple[Optional[int], str]:
    comp_board = comp_e[comp_e["Type of Committee"].str.lower() == "board of directors"]
    size1 = len(_filter_directors(comp_board))
    if size1:
        return size1, "from Sheet1 ('Board of Directors')"
    if not board_fy.empty and "Total No. of Directors Present in the Meeting" in board_fy.columns:
        observed = pd.to_numeric(board_fy["Total No. of Directors Present in the Meeting"], errors="coerce")
        if observed.notna().any():
            return int(observed.max()), "derived from max directors present in Sheet3"
    return None, "unavailable"


def evaluate_board_meetings(comp_e_fy: pd.DataFrame, board_fy: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    if board_fy.empty:
        summary = pd.DataFrame(
            [
                {"Rule": "Board meetings in FY", "Expected": thresholds.BOARD_MIN_MEETINGS, "Observed/Status": f"0 ({status.FAIL_TAG})"},
                {"Rule": "Quorum per meeting", "Expected": f"max({thresholds.BOARD_QUORUM_MIN}, ceil(BoardSize/3))", "Observed/Status": "—"},
                {"Rule": "Independent Director present (each mtg)", "Expected": f"≥ {thresholds.BOARD_MIN_INDEPENDENT_PRESENT}", "Observed/Status": "—"},
                {"Rule": "Max gap between meetings (days)", "Expected": thresholds.BOARD_MAX_GAP_DAYS, "Observed/Status": "—"},
            ]
        )
        return summary, pd.DataFrame(), False

    board = _sorted_meetings(board_fy, "Date of Board Meeting")

    meet_cnt = len(board)
    freq_ok = meet_cnt >= thresholds.BOARD_MIN_MEETINGS

    gap_ok, worst_gap = _max_gap_check(board["Meeting Date"], thresholds.BOARD_MAX_GAP_DAYS)

    board_size, prov = _board_size_from_sheet1_or_observed(comp_e_fy, board)
    quorum_needed = max(thresholds.BOARD_QUORUM_MIN, math.ceil(board_size / 3)) if board_size else None

    checks, all_meets_ok = _check_each_meeting(
        board, "Total No. of Directors Present in the Meeting", quorum_needed, thresholds.BOARD_MIN_INDEPENDENT_PRESENT
    )
    per_table = pd.DataFrame([
        {
            "Date": c["date"],
            "Directors Present": c["present"],
            "Independent Present": c["indep_present"],
            "Quorum Needed": quorum_needed if quorum_needed is not None else f"n/a ({prov})",
            "Quorum OK": _pass_fail(c["q_ok"]),
            "≥1 ID Present": _pass_fail(c["id_ok"]),
        }
        for c in checks
    ])

    summary = pd.DataFrame(
        [
            {"Rule": "Board meetings in FY", "Expected": thresholds.BOARD_MIN_MEETINGS, "Observed/Status": f"{meet_cnt} ({_pass_fail(freq_ok)})"},
            {
                "Rule": "Quorum per meeting",
                "Expected": f"max({thresholds.BOARD_QUORUM_MIN}, ceil(BoardSize/3)) [{prov}]",
                "Observed/Status": _all_pass_or(per_table, "Quorum OK", "Some meetings fail quorum"),
            },
            {
                "Rule": "Independent Director present (each mtg)",
                "Expected": f"≥ {thresholds.BOARD_MIN_INDEPENDENT_PRESENT}",
                "Observed/Status": _all_pass_or(per_table, "≥1 ID Present", "Some meetings lack IDs"),
            },
            {"Rule": "Max gap between meetings (days)", "Expected": thresholds.BOARD_MAX_GAP_DAYS,
             "Observed/Status": _gap_status(gap_ok, worst_gap, thresholds.BOARD_MAX_GAP_DAYS)},
        ]
    )
    all_ok = freq_ok and gap_ok and all_meets_ok
    return summary, per_table, all_ok


def evaluate_independent_directors_meeting_sheet4(ind_fy: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Sheet4 structure:
      - Date of Meeting of Independent Directors
      - Total No. of Independent directors in the meeting
      - (and other counts)
    Rule: at least 1 meeting in the FY.
    """
    if ind_fy.empty:
        return pd.DataFrame([{"Rule": "Independent Directors’ meeting in FY", "Expected": thresholds.INDEPENDENT_DIRECTORS_MIN_MEETINGS, "Observed/Status": f"0 ({status.FAIL_TAG})"}]), pd.DataFrame(), False

    df = ind_fy.copy()
    if "Date of Meeting of Independent Directors" not in df.columns:
        return pd.DataFrame([{
            "Rule": "Independent Directors’ meeting in FY",
            "Expected": thresholds.INDEPENDENT_DIRECTORS_MIN_MEETINGS,
            "Observed/Status": "— (Sheet4 date column not found)"
        }]), pd.DataFrame(), False

    df["Meeting Date"] = _parse_meeting_dates(df["Date of Meeting of Independent Directors"])
    meetings = df[df["Meeting Date"].notna()].sort_values("Meeting Date")

    count = len(meetings)
    ok = count >= thresholds.INDEPENDENT_DIRECTORS_MIN_MEETINGS

    per_rows = []
    for _, r in meetings.iterrows():
        per_rows.append(
            {
                "Date": r["Meeting Date"].date(),
                "Independent Present": pd.to_numeric(r.get("Total No. of Independent directors in the meeting", ""), errors="coerce"),
                "Non-Independent/Other Present": pd.to_numeric(r.get("Total No. of Non-Independent directors/Other Members in the meeting", ""), errors="coerce"),
                "IDs via AV Means": pd.to_numeric(r.get("Total No. of Independent directors Joining by Audio-Visual Means", ""), errors="coerce"),
            }
        )
    per_table = pd.DataFrame(per_rows) if per_rows else pd.DataFrame()

    summary = pd.DataFrame([{
        "Rule": "Independent Directors’ meeting in FY",
        "Expected": thresholds.INDEPENDENT_DIRECTORS_MIN_MEETINGS,
        "Observed/Status": f"{count} ({status.PASS_TAG if ok else status.FAIL_TAG})"
    }])
    return summary, per_table, ok


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
def render() -> None:
    st.title("Governance — Committee Composition Checks")

    seg = st.sidebar.selectbox("Select Segment", ["REIT"], index=0)
    
    # Auto-select URL (Hidden)
    url = DEFAULT_GOVERNANCE_URL

    comp, meetings, board, ind, notes = _load_all_sheets(url)
    if notes:
        for n in notes:
            st.info(n)

    if comp.empty:
        st.warning("No rows loaded from the composition sheet.")
        return

    # Validate Sheet1
    required_comp = [
        "Name of REIT", "Financial Year", "Period Ended", "Type of Committee",
        "Name of Member of Committee", "Type of Members of Committee", "Role of Members of Committee",
        "Is this member identified as having accounting or related Financial Management Expertise.",
        "Is this Member the Chairperson for the Committee",
    ]
    missing = [c for c in required_comp if c not in comp.columns]
    if missing:
        st.error(f"Missing columns in Sheet1 (composition): {missing}")
        st.dataframe(comp.head())
        return
    comp = comp.map(_clean_str)

    # Validate Sheet2 (optional)
    meetings_ok = False
    if not meetings.empty:
        need2 = [
            "Name of REIT","Financial Year","Period Ended","Type of Committee",
            "Date of Meeting of Committee","Total No. of Members Present in the Meeting",
            "Total No. of Independent directors in the meeting",
        ]
        miss2 = [c for c in need2 if c not in meetings.columns]
        if miss2:
            st.warning(f"Sheet2 missing columns: {miss2}. Meeting checks disabled.")
        else:
            meetings = meetings.map(_clean_str)
            meetings_ok = True

    # Validate Sheet3 (optional)
    board_ok = False
    if not board.empty:
        need3 = [
            "Name of REIT","Financial Year","Period Ended","Date of Board Meeting",
            "Total No. of Directors Present in the Meeting","Total No. of Independent directors in the meeting",
        ]
        miss3 = [c for c in need3 if c not in board.columns]
        if miss3:
            st.warning(f"Sheet3 missing columns: {miss3}. Board checks may be partial.")
        board = board.map(_clean_str)
        board_ok = True

    # Validate Sheet4 (optional)
    ind_ok = False
    if not ind.empty:
        need4 = ["Name of REIT","Financial Year","Period Ended","Date of Meeting of Independent Directors"]
        miss4 = [c for c in need4 if c not in ind.columns]
        if miss4:
            st.warning(f"Sheet4 missing columns: {miss4}. Independent Directors’ check disabled.")
        else:
            ind = ind.map(_clean_str)
            ind_ok = True

    # Selections
    entity = st.sidebar.selectbox("Choose REIT", sorted(comp["Name of REIT"].unique()))
    comp_e = comp[comp["Name of REIT"] == entity]

    years = sorted(comp_e["Financial Year"].unique())
    fy = st.sidebar.selectbox("Financial Year", years, index=max(0, len(years) - 1))
    comp_ey = comp_e[comp_e["Financial Year"] == fy]

    periods = sorted(comp_ey["Period Ended"].unique())
    period = st.sidebar.selectbox("Period Ended", periods, index=0)
    comp_now = comp_ey[comp_ey["Period Ended"] == period]

    # Sheet2 subset for FY
    meetings_fy = pd.DataFrame()
    if meetings_ok:
        meetings_fy = meetings[(meetings["Name of REIT"] == entity) & (meetings["Financial Year"] == fy)].copy()

    # Sheet3 subset for FY
    board_fy = pd.DataFrame()
    if board_ok:
        board_fy = board[(board["Name of REIT"] == entity) & (board["Financial Year"] == fy)].copy()

    # Sheet4 subset for FY
    ind_fy = pd.DataFrame()
    if ind_ok:
        ind_fy = ind[(ind["Name of REIT"] == entity) & (ind["Financial Year"] == fy)].copy()

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

        if meetings_fy.empty:
            st.info("Meeting sheet not available or no rows for this REIT/FY.")
            continue

        st.markdown(f"**Meetings — {title} (FY {fy})**")
        summary, per_meeting, ok = evaluate_meetings_for_committee(comp_now, meetings_fy, title)
        st.table(summary)
        if not per_meeting.empty:
            st.dataframe(per_meeting, width="stretch")
        if ok:
            st.success("All meeting rules satisfied for the FY (given the selected period's committee size).")
        else:
            st.error("One or more meeting rules are not satisfied for the FY.")

    # ----- Board of Directors (Sheet3)
    st.subheader("Board of Directors — Meetings (FY level)")
    if board_fy.empty:
        st.info("Sheet3 not available or no Board rows for this REIT/FY.")
    else:
        summary_b, per_b, ok_b = evaluate_board_meetings(comp_ey, board_fy)
        st.table(summary_b)
        if not per_b.empty:
            st.dataframe(per_b, width="stretch")
        if ok_b:
            st.success("All Board meeting rules satisfied for the FY.")
        else:
            st.error("One or more Board meeting rules are not satisfied for the FY.")

    # ----- Independent Directors’ Meeting (Sheet4)
    st.subheader("Independent Directors — Meeting (FY level)")
    if ind_fy.empty:
        st.info("Sheet4 not available or no Independent Directors rows for this REIT/FY.")
    else:
        id_summary, id_per, id_ok = evaluate_independent_directors_meeting_sheet4(ind_fy)
        st.table(id_summary)
        if not id_per.empty:
            st.dataframe(id_per, width="stretch")
        if id_ok:
            st.success("Independent Directors met at least once in the FY.")
        else:
            st.error("Independent Directors did NOT meet at least once in the FY.")

def render_governance() -> None:
    render()
