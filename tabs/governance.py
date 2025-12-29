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


def _load_comp_and_meetings(url: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads Sheet1 (composition) and Sheet2 (meetings). Returns (comp, meetings, warnings).
    """
    notes: List[str] = []
    # Try to read by sheet name, fall back to default-first-sheet if needed.
    try:
        comp = read_google_sheet_by_sheetname(url, "Sheet1")
    except Exception:
        comp = read_google_sheet_csv_default(url)
        notes.append("Sheet1 by name failed; used first sheet as composition.")

    # Meetings (Sheet2)
    meetings = pd.DataFrame()
    try:
        meetings = read_google_sheet_by_sheetname(url, "Sheet2")
    except Exception:
        notes.append("Sheet2 not found by name; meeting checks will be skipped.")

    return comp, meetings, notes


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
# Meeting rules / evaluation (Sheet2)
# --------------------------------------------------------------------
_MEETING_RULES: Dict[str, Dict[str, Optional[int]]] = {
    # per FY minimum meetings; gap thresholds in days (None = no rule)
    "Audit Committee": {"min_meetings": 4, "gap_days": 120, "min_indep_present": 2},
    "Nomination and Remuneration Committee": {"min_meetings": 1, "gap_days": None, "min_indep_present": 1},
    "Stakeholders Relationship Committee": {"min_meetings": 1, "gap_days": None, "min_indep_present": 1},
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
    """
    rules = _MEETING_RULES[committee]
    size = _committee_size_from_comp(comp_now, committee)
    quorum_needed = max(2, math.ceil(size / 3)) if size else None
    min_indep = rules["min_indep_present"]

    # Subset meetings of this committee in the FY
    m = meetings_fy[meetings_fy["Type of Committee"].str.lower() == committee.lower()].copy()
    if m.empty:
        # No meetings in FY — build a red summary with what was expected
        out_summary = pd.DataFrame(
            {
                "Rule": [
                    "Meetings in FY",
                    "Quorum per meeting",
                    "Min independent directors per meeting",
                    "Max gap between meetings (days)",
                ],
                "Expected": [
                    rules["min_meetings"],
                    quorum_needed if quorum_needed is not None else "n/a (no committee size)",
                    min_indep,
                    rules["gap_days"] if rules["gap_days"] is not None else "—",
                ],
                "Observed/Status": ["0 (🔴)", "—", "—", "—"],
            }
        )
        return out_summary, pd.DataFrame(), False

    # Parse dates
    m["Meeting Date"] = _parse_meeting_dates(m["Date of Meeting of Committee"])
    m = m.sort_values("Meeting Date")

    # Frequency rule (FY)
    meet_cnt = len(m)
    freq_ok = meet_cnt >= int(rules["min_meetings"])

    # Gap rule (FY)
    gap_days_rule = rules["gap_days"]
    gap_ok = True
    worst_gap = None
    if gap_days_rule is not None and meet_cnt >= 2:
        diffs = (m["Meeting Date"].diff().dt.days).iloc[1:]  # skip first NaN
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

        id_needed = min_indep if min_indep is not None else "-"
        id_ok = (indep_present is not None and min_indep is not None and indep_present >= min_indep)

        per_rows.append(
            {
                "Date": row["Meeting Date"].date() if pd.notna(row["Meeting Date"]) else "—",
                "Members Present": present if present is not None else "—",
                "Independent Present": indep_present if indep_present is not None else "—",
                "Quorum Needed": q_needed,
                "Quorum OK": "🟢" if q_ok else "🔴",
                "Independents Needed": id_needed,
                "IDs OK": "🟢" if id_ok else "🔴",
            }
        )
        all_meets_ok = all_meets_ok and q_ok and id_ok

    per_table = pd.DataFrame(per_rows)

    # Build summary
    gap_text = (
        f"Worst gap: {worst_gap} (OK≤{gap_days_rule})" if worst_gap is not None else ("—" if gap_days_rule is None else "n/a")
    )
    summary = pd.DataFrame(
        {
            "Rule": [
                "Meetings in FY",
                "Quorum per meeting",
                "Min independent directors per meeting",
                "Max gap between meetings (days)",
            ],
            "Expected": [
                rules["min_meetings"],
                quorum_needed if quorum_needed is not None else "n/a (no committee size)",
                min_indep,
                gap_days_rule if gap_days_rule is not None else "—",
            ],
            "Observed/Status": [
                f"{meet_cnt} ({'🟢' if freq_ok else '🔴'})",
                "OK" if per_table["Quorum OK"].eq("🟢").all() else "🔴 Some meetings fail quorum",
                "OK" if per_table["IDs OK"].eq("🟢").all() else "🔴 Some meetings lack IDs",
                ("🟢 " + gap_text) if gap_ok else ("🔴 " + gap_text),
            ],
        }
    )

    all_ok = freq_ok and gap_ok and all_meets_ok
    return summary, per_table, all_ok


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

    comp, meetings, notes = _load_comp_and_meetings(url)
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

    # Optional meetings sheet validations
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

    # Selections
    entity = st.selectbox("Choose REIT", sorted(comp["Name of REIT"].unique()))
    comp_e = comp[comp["Name of REIT"] == entity]

    years = sorted(comp_e["Financial Year"].unique())
    fy = st.selectbox("Financial Year", years, index=max(0, len(years) - 1))
    comp_ey = comp_e[comp_e["Financial Year"] == fy]

    periods = sorted(comp_ey["Period Ended"].unique())
    period = st.selectbox("Period Ended", periods, index=0)
    comp_now = comp_ey[comp_ey["Period Ended"] == period]

    # Meetings subset for this entity & FY (not filtered by period – frequency/gap are FY-level)
    meetings_fy = pd.DataFrame()
    if meetings_ok:
        meetings_fy = meetings[(meetings["Name of REIT"] == entity) & (meetings["Financial Year"] == fy)].copy()

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

def render_governance() -> None:
    render()
