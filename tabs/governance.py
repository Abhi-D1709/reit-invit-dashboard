# tabs/governance.py
import re
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Defaults / wiring (tries utils.common first; falls back to your URL)
# --------------------------------------------------------------------
DEFAULT_GOVERNANCE_URL = (
    "https://docs.google.com/spreadsheets/d/1ETx5UZKQQyZKxkF4fFJ4R9wa7i7TNp7EXIhHWiVYG7s/edit?usp=sharing"
)
try:
    # If you later add GOVERNANCE_REIT_SHEET_URL in utils/common.py
    from utils.common import GOVERNANCE_REIT_SHEET_URL  # type: ignore
    if isinstance(GOVERNANCE_REIT_SHEET_URL, str) and GOVERNANCE_REIT_SHEET_URL.strip():
        DEFAULT_GOVERNANCE_URL = GOVERNANCE_REIT_SHEET_URL.strip()
except Exception:
    pass


# ------------------------------- Utilities -----------------------------------
def _csv_url_from_gsheet(url: str, *, sheet: Optional[str] = None, gid: Optional[str] = None) -> str:
    """
    Build a CSV URL for a public Google Sheet.
    - If gid is given, export that grid as CSV.
    - Else if sheet is given, use the gviz CSV API for that sheet name.
    - Else export the first sheet as CSV.
    """
    m = re.search(r"/d/([a-zA-Z0-9\-_]+)", url)
    if not m:
        return url
    sheet_id = m.group(1)
    if gid:
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    if sheet:
        from urllib.parse import quote
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet)}"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"


def _strip(s):
    return str(s).strip() if pd.notna(s) else s


def _status(v: Optional[bool]) -> str:
    if pd.isna(v):
        return "—"
    return "🟢" if bool(v) else "🔴"


def _norm_bool(x) -> Optional[bool]:
    """Normalize Yes/No/True/False/Y/N/1/0 to Python bool, else None."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "t", "1"}:
        return True
    if s in {"no", "n", "false", "f", "0"}:
        return False
    return None


def _contains(s: str, token: str) -> bool:
    return token.lower() in (s or "").lower()


# ------------------------------- Data loader ---------------------------------
def _read_governance_df(sheet_url: str) -> pd.DataFrame:
    """
    Reads the governance sheet and returns a normalized dataframe.
    Expected columns (case/spacing tolerant):
      - Name of REIT
      - Financial Year
      - Period Ended
      - Type of Committee
      - Name of Member of Committee
      - Type of Members of Committee        (expects e.g., 'Independent Director', 'Non-Independent Director')
      - Role of Members of Committee        (expects e.g., 'Non-Executive', 'Executive')
      - Is this member identified as having accounting or related Financial Management Expertise.
      - Is this Member the Chairperson for the Committee
      - Is this Member part of any committee in any other listed entity  (unused in checks, retained for display)
    """
    # Most workbooks keep this data in the first sheet; export that:
    csv_url = _csv_url_from_gsheet(sheet_url or DEFAULT_GOVERNANCE_URL)
    df = pd.read_csv(csv_url, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Column normalization (be tolerant to small header variations)
    rename_map = {
        "Entity": "Name of REIT",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Type of Members of the Committee": "Type of Members of Committee",
        "Role of Members of the Committee": "Role of Members of Committee",
        "Is this member identified as having accounting or related financial management expertise":
            "Is this member identified as having accounting or related Financial Management Expertise.",
        "Is the Member the Chairperson for the Committee":
            "Is this Member the Chairperson for the Committee",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

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
        st.error("Governance sheet is missing columns: " + ", ".join(missing))
        with st.expander("Show detected columns"):
            st.write(list(df.columns))
        return df.iloc[0:0]

    # Clean string columns
    for c in ["Name of REIT", "Financial Year", "Period Ended", "Type of Committee",
              "Name of Member of Committee", "Type of Members of Committee", "Role of Members of Committee"]:
        df[c] = df[c].astype(str).map(_strip)

    # Normalize booleans
    df["Has FM Expertise"] = df["Is this member identified as having accounting or related Financial Management Expertise."].map(_norm_bool)
    df["Is Chair"] = df["Is this Member the Chairperson for the Committee"].map(_norm_bool)

    return df


# ------------------------------ Checks logic ---------------------------------
def _summarize_committee(df: pd.DataFrame) -> Dict[str, Tuple[bool, str]]:
    """
    Given rows for a REIT + FY + Period + Committee, compute rules and
    return mapping {rule_label: (pass_bool, human_note)}
    """
    n = len(df)
    indep = sum(_contains(t, "independent") for t in df["Type of Members of Committee"])
    non_exec_all = all(_contains(r, "non-executive") for r in df["Role of Members of Committee"])
    has_fm = any(df["Has FM Expertise"].fillna(False))

    # Chair row (if any)
    chair_rows = df[df["Is Chair"] == True]
    chair_type = chair_rows["Type of Members of Committee"].iloc[0] if not chair_rows.empty else None
    chair_role = chair_rows["Role of Members of Committee"].iloc[0] if not chair_rows.empty else None

    def _ratio_fmt(a, b) -> str:
        return f"{a}/{b} ({(a/b*100):.0f}%)" if b else "0/0"

    rules: Dict[str, Tuple[bool, str]] = {}

    committee = (df["Type of Committee"].iloc[0] or "").strip().lower()

    # Common counters
    at_least_three = n >= 3
    two_thirds_indep = (indep / n >= 2/3) if n else False

    # Committee-specific rules
    if "audit" in committee:
        rules["Min 3 directors"] = (at_least_three, f"Members: {n}")
        rules["≥ 2/3 independent"] = (two_thirds_indep, f"Independent: {_ratio_fmt(indep, n)}")
        rules["≥ 1 member has financial expertise"] = (has_fm, "Yes" if has_fm else "No")
        chair_indep = _contains(chair_type or "", "independent")
        rules["Chairperson is independent"] = (chair_indep, chair_type or "—")

    elif "nomination" in committee or "remuneration" in committee:
        rules["Min 3 directors"] = (at_least_three, f"Members: {n}")
        rules["All directors are non-executive"] = (non_exec_all, "All Non-Executive" if non_exec_all else "Mixed")
        rules["≥ 2/3 independent"] = (two_thirds_indep, f"Independent: {_ratio_fmt(indep, n)}")
        chair_indep = _contains(chair_type or "", "independent")
        rules["Chairperson is independent"] = (chair_indep, chair_type or "—")

    elif "stakeholder" in committee:
        chair_non_exec = _contains(chair_role or "", "non-executive")
        rules["Chairperson is non-executive"] = (chair_non_exec, chair_role or "—")
        rules["Min 3 directors"] = (at_least_three, f"Members: {n}")
        rules["≥ 1 independent director"] = (indep >= 1, f"Independent: {indep}")

    elif "risk" in committee:
        rules["Min 3 directors"] = (at_least_three, f"Members: {n}")
        rules["≥ 1 independent director"] = (indep >= 1, f"Independent: {indep}")

    else:
        # Unknown committee – no rules but show counts, so the user can see data
        rules["Members (info)"] = (True, f"{n}")
        rules["Independent (info)"] = (True, f"{indep}")
        if chair_type or chair_role:
            rules["Chair (info)"] = (True, f"{chair_type or ''} / {chair_role or ''}")

    return rules


def _render_rules_table(title: str, rules: Dict[str, Tuple[bool, str]]) -> None:
    st.subheader(title)
    if not rules:
        st.info("No data for this committee and period.")
        return
    df = pd.DataFrame(
        [{"Check": k, "Result": _status(v[0]), "Detail": v[1]} for k, v in rules.items()]
    )
    st.dataframe(df, hide_index=True, use_container_width=True)
    if any(v[0] is False for v in rules.values()):
        st.error("One or more checks failed for this committee/period.")


# ---------------------------------- UI ---------------------------------------
def render():
    st.header("Governance — Committee Composition Checks")

    # Sidebar controls (match the overall app style)
    with st.sidebar:
        seg = st.selectbox("Select Segment", ["REIT", "InvIT"], index=0)
        sheet_url = st.text_input(
            "Data URL (Google Sheet - public view)",
            value=DEFAULT_GOVERNANCE_URL,
            help="Public Google Sheet containing committee composition rows.",
        )

    if seg != "REIT":
        st.info("InvIT governance checks will be added later.")
        return

    df_all = _read_governance_df(sheet_url)
    if df_all.empty:
        return

    # Top-level filters: REIT → FY → Period
    ent = st.selectbox("Choose REIT", sorted(df_all["Name of REIT"].dropna().unique().tolist()))
    df_ent = df_all[df_all["Name of REIT"] == ent].copy()
    fy = st.selectbox(
        "Financial Year",
        sorted(df_ent["Financial Year"].dropna().unique().tolist()),
    )
    df_fy = df_ent[df_ent["Financial Year"] == fy].copy()
    period = st.selectbox(
        "Period Ended",
        sorted(df_fy["Period Ended"].dropna().unique().tolist()),
    )

    df_q = df_fy[df_fy["Period Ended"] == period].copy()
    if df_q.empty:
        st.warning("No rows found for the selected REIT / FY / Period.")
        return

    # Work committee by committee for the selected slice
    committees = (
        df_q["Type of Committee"].dropna().astype(str).map(_strip).unique().tolist()
    )
    for comm in sorted(committees):
        rules = _summarize_committee(df_q[df_q["Type of Committee"] == comm].copy())
        _render_rules_table(comm, rules)
