# tabs/governance.py
from __future__ import annotations

import re
from typing import Tuple, List

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
# Helpers: cleaners, classifiers, and Google Sheet reader
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
    Robust independence classifier:
      - values that START with 'non independent' (with/without hyphen) => NOT independent
      - values that START with 'independent' => independent
      - ignores case/hyphens/extra spaces
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
    Treat as 'director' only when the type explicitly contains 'director'
    (e.g., 'Independent Director', 'Non-Independent Director', 'Executive Director').
    This excludes CXO/CEO/Company Secretary/etc.
    """
    s = _clean_str(type_cell).lower()
    return "director" in s


def _filter_directors(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Type of Members of Committee"].apply(_is_director)].copy()


@st.cache_data(show_spinner=False)
def read_google_sheet_public(url: str) -> pd.DataFrame:
    """
    Reads a publicly shared Google Sheet (viewer link) by swapping to CSV export.
    """
    url = _clean_str(url)
    if "/edit" in url:
        base = url.split("/edit")[0]
    elif "/view" in url:
        base = url.split("/view")[0]
    else:
        base = url.rstrip("/")
    csv_url = f"{base}/export?format=csv"
    df = pd.read_csv(csv_url, dtype=str).applymap(_clean_str)
    return df


# --------------------------------------------------------------------
# Core evaluation per-committee  (DIRECTOR-ONLY counting where relevant)
# --------------------------------------------------------------------
def evaluate_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit Committee checks:
      1. minimum 3 directors;
      2. at least 2/3 should be independent directors;
      3. at least 1 should have financial management expertise
      4. Chairperson should be independent director
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

    # Chair: must exist AND be a director AND be independent
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
    Nomination & Remuneration Committee:
      1. Minimum 3 directors
      2. All the directors – non-executive
      3. At least 2/3 should be independent
      4. Chairperson should be independent director
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
    Stakeholders Relationship Committee:
      1. Chairperson should be non-executive
      2. Minimum 3 directors
      3. Minimum 1 independent director
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
    Risk Management Committee:
      1. Minimum 3 directors
      2. Minimum 1 independent director
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
# UI
# --------------------------------------------------------------------
def render() -> None:
    st.title("Governance — Committee Composition Checks")

    seg = st.sidebar.selectbox("Select Segment", ["REIT"], index=0)
    _ = seg  # reserved for future InvIT support

    url = st.sidebar.text_input(
        "Data URL (Google Sheet - public view)", value=DEFAULT_GOVERNANCE_URL
    )

    df = read_google_sheet_public(url)
    if df.empty:
        st.warning("No rows loaded from the provided Google Sheet.")
        return

    required = [
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in sheet: {missing}")
        st.dataframe(df.head())
        return

    df = df.applymap(_clean_str)

    entity = st.selectbox("Choose REIT", sorted(df["Name of REIT"].unique()))
    df_e = df[df["Name of REIT"] == entity]

    years = sorted(df_e["Financial Year"].unique())
    fy = st.selectbox("Financial Year", years, index=max(0, len(years) - 1))
    df_ey = df_e[df_e["Financial Year"] == fy]

    periods = sorted(df_ey["Period Ended"].unique())
    period = st.selectbox("Period Ended", periods, index=0)
    df_now = df_ey[df_ey["Period Ended"] == period]

    committees = {
        "Audit Committee": evaluate_audit,
        "Nomination and Remuneration Committee": evaluate_nrc,
        "Stakeholders Relationship Committee": evaluate_src,
        "Risk Management Committee": evaluate_rmc,
    }

    for title, fn in committees.items():
        st.subheader(title)
        sub = df_now[df_now["Type of Committee"].str.lower() == title.lower()]
        st.table(fn(sub))


def render_governance() -> None:
    render()
