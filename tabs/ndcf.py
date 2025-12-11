import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Optional

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NDCF", layout="wide")

# Defaults; can be overridden by utils.common if present
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing"
DEFAULT_SHEET_NAME = "NDCF REITs"

try:
    # Optional: use the central URL if you added it in utils/common.py
    from utils.common import NDCF_REITS_SHEET_URL  # type: ignore
    if NDCF_REITS_SHEET_URL:
        DEFAULT_SHEET_URL = NDCF_REITS_SHEET_URL
except Exception:
    pass


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _csv_url_from_gsheet(url: str, sheet: Optional[str] = None, gid: Optional[str] = None) -> str:
    """
    Build a CSV export URL for a Google Sheet. Prefer the sheet-name route,
    fall back to gid if provided.
    """
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        return url
    sheet_id = m.group(1)
    if gid:
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    if sheet:
        from urllib.parse import quote
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet)}"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"


def _to_number(x):
    """Parse numbers like '1,234.56', '-', '', '(123.45)' into floats/NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s in {"", "-", "–", "—"}:
        return np.nan
    s = s.replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return np.nan


# -----------------------------------------------------------------------------
# Data load & checks
# -----------------------------------------------------------------------------
def load_reit_ndcf(url: str, sheet_name: str = DEFAULT_SHEET_NAME) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]

    # Harmonise common variants/typos
    rename_map = {
        "Entity": "Name of REIT",
        "Fincial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Period Ended": "Period Ended",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    needed = [
        "Name of REIT",
        "Financial Year",
        "Period Ended",
        "Total Amount of NDCF computed as per NDCF Statement",
        "Total Amount of NDCF declared for the period (incl. Surplus)",
        "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
        "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
        "Cash Flow From Fincing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
        "Profit after tax as per Statement of Profit and Loss (as per Audited Fincials or Fincials with Limited Review)",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error("The NDCF sheet does not have the expected columns. Missing: " + ", ".join(missing))
        with st.expander("Show detected columns"):
            st.write(list(df.columns))
        return df.iloc[0:0]

    # Numeric conversions
    for c in needed[3:]:
        df[c] = df[c].map(_to_number)

    # Clean text fields
    for c in ["Name of REIT", "Financial Year", "Period Ended"]:
        df[c] = df[c].astype(str).str.strip()

    return df


def compute_checks(df: pd.DataFrame) -> pd.DataFrame:
    comp = "Total Amount of NDCF computed as per NDCF Statement"
    decl = "Total Amount of NDCF declared for the period (incl. Surplus)"
    cfo = "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)"
    cfi = "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)"
    cff = "Cash Flow From Fincing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)"
    pat = "Profit after tax as per Statement of Profit and Loss (as per Audited Fincials or Fincials with Limited Review)"

    out = df.copy()
    out["Payout Ratio %"] = np.where(out[comp] > 0, (out[decl] / out[comp]) * 100.0, np.nan).round(2)
    out["Meets 90% Rule"] = out["Payout Ratio %"] >= 90.0

    out["CF Sum"] = out[cfo].fillna(0) + out[cfi].fillna(0) + out[cff].fillna(0) + out[pat].fillna(0)
    out["Gap vs Computed"] = out["CF Sum"] - out[comp]
    out["Gap % of Computed"] = np.where(out[comp] != 0, (out["Gap vs Computed"] / out[comp]) * 100.0, np.nan).round(2)
    out["Within 10% Gap"] = out["Gap % of Computed"].abs() <= 10.0
    return out


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def render():
    st.header("NDCF — Compliance Checks")

    with st.sidebar:
        seg = st.selectbox("Select Segment", ["REIT", "InvIT"], index=0)
        data_url = st.text_input(
            "Data URL (Google Sheet - public view)",
            value=DEFAULT_SHEET_URL,
            help="Expected sheet name: 'NDCF REITs'.",
        )

    if seg != "REIT":
        st.info("InvIT checks will be added later.")
        return

    df_all = load_reit_ndcf(data_url, DEFAULT_SHEET_NAME)
    if df_all.empty:
        return

    c1, c2 = st.columns([1.2, 0.6])
    with c1:
        ent = st.selectbox("Choose REIT", sorted(df_all["Name of REIT"].dropna().unique().tolist()))
    with c2:
        fy_options = ["All"] + sorted(df_all.loc[df_all["Name of REIT"] == ent, "Financial Year"].dropna().unique().tolist())
        fy = st.selectbox("Financial Year", fy_options, index=0)

    q = df_all[df_all["Name of REIT"] == ent].copy()
    if fy != "All":
        q = q[q["Financial Year"] == fy]

    q = compute_checks(q)

    # Summary
    total = int(len(q))
    good_payout = int(q["Meets 90% Rule"].sum())
    good_gap = int(q["Within 10% Gap"].sum())

    s1, s2, s3 = st.columns(3)
    s1.metric("Periods meeting 90% payout", f"{good_payout}/{total}")
    s2.metric("Periods within 10% gap", f"{good_gap}/{total}")
    s3.metric("Rows analysed", f"{total}")

    def status_icon(v: Optional[bool]) -> str:
        if pd.isna(v):
            return "—"
        return "🟢" if bool(v) else "🔴"

    disp = q[[
        "Financial Year",
        "Period Ended",
        "Total Amount of NDCF computed as per NDCF Statement",
        "Total Amount of NDCF declared for the period (incl. Surplus)",
        "Payout Ratio %",
        "Meets 90% Rule",
        "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
        "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
        "Cash Flow From Fincing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
        "Profit after tax as per Statement of Profit and Loss (as per Audited Fincials or Fincials with Limited Review)",
        "CF Sum",
        "Gap vs Computed",
        "Gap % of Computed",
        "Within 10% Gap",
    ]].copy()

    disp["Meets 90% Rule"] = disp["Meets 90% Rule"].map(status_icon)
    disp["Within 10% Gap"] = disp["Within 10% Gap"].map(status_icon)

    st.write("### Period-wise results")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Alerts
    if (~q["Meets 90% Rule"].fillna(False)).any():
        st.error("One or more periods do **not** meet the 90% payout requirement (Declared incl. surplus < 90% of Computed NDCF).")
    if (~q["Within 10% Gap"].fillna(False)).any():
        st.error("One or more periods have a gap **> 10%** between (CFO + CFI + CFF + PAT) and Computed NDCF.")


# For pages/4_NDCF.py
if __name__ == "__main__":
    render()
