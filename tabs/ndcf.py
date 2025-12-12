# tabs/ndcf.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Optional

st.set_page_config(page_title="NDCF", layout="wide")

# ---- Config (overridden by utils.common if present) -------------------------
DEFAULT_SHEET_URL_TRUST = "https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing"
TRUST_SHEET_NAME = "NDCF REITs"
SPV_SHEET_NAME   = "NDCF SPV REIT"

try:
    from utils.common import NDCF_REITS_SHEET_URL  # optional central location
    if NDCF_REITS_SHEET_URL:
        DEFAULT_SHEET_URL_TRUST = NDCF_REITS_SHEET_URL
except Exception:
    pass

# ---- Helpers ----------------------------------------------------------------
def _csv_url_from_gsheet(url: str, sheet: Optional[str] = None, gid: Optional[str] = None) -> str:
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

def _strip(s):
    return str(s).strip() if pd.notna(s) else s

def _fails(series: pd.Series) -> pd.Series:
    """Return boolean mask of failures, safely handling NaN/object dtypes."""
    return series.astype("boolean").fillna(False).eq(False)

def _status(v: Optional[bool]) -> str:
    if pd.isna(v):
        return "—"
    return "🟢" if bool(v) else "🔴"

# ---- TRUST-LEVEL -------------------------------------------------------------
def load_reit_ndcf(url: str, sheet_name: str = TRUST_SHEET_NAME) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Entity": "Name of REIT",
        "Fincial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
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
        st.error("The NDCF (Trust) sheet is missing expected columns: " + ", ".join(missing))
        with st.expander("Show detected columns (Trust)"):
            st.write(list(df.columns))
        return df.iloc[0:0]

    for c in needed[3:]:
        df[c] = df[c].map(_to_number)
    for c in ["Name of REIT", "Financial Year", "Period Ended"]:
        df[c] = df[c].astype(str).map(_strip)
    return df

def compute_trust_checks(df: pd.DataFrame) -> pd.DataFrame:
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

# ---- SPV-LEVEL ---------------------------------------------------------------
def load_reit_spv_ndcf(url: str, sheet_name: str = SPV_SHEET_NAME) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Entity": "Name of REIT",
        "Fincial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Name of Holdco": "Name of Holdco (Leave Blank if N/A)",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    needed = [
        "Name of REIT",
        "Name of SPV",
        "Name of Holdco (Leave Blank if N/A)",
        "Financial Year",
        "Period Ended",
        "Total Amount of NDCF computed as per NDCF Statement",
        "Total Amount of NDCF declared for the period (incl. Surplus)",
        "SPV Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "SPV Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "SPV Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "SPV Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)",
        "HoldCo Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "HoldCo Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "Holdco Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "Holdco Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning("The NDCF (SPV) sheet is missing expected columns: " + ", ".join(missing))
        with st.expander("Show detected columns (SPV)"):
            st.write(list(df.columns))
        return df.iloc[0:0]

    for c in needed[5:]:
        df[c] = df[c].map(_to_number)
    for c in ["Name of REIT", "Financial Year", "Period Ended", "Name of SPV", "Name of Holdco (Leave Blank if N/A)"]:
        df[c] = df[c].astype(str).map(_strip)
    return df

def compute_spv_checks(df: pd.DataFrame) -> pd.DataFrame:
    comp = "Total Amount of NDCF computed as per NDCF Statement"
    decl = "Total Amount of NDCF declared for the period (incl. Surplus)"

    spv_cfo = "SPV Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    spv_cfi = "SPV Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    spv_cff = "SPV Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    spv_pat = "SPV Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"

    hco_cfo = "HoldCo Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    hco_cfi = "HoldCo Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    hco_cff = "Holdco Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    hco_pat = "Holdco Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"

    out = df.copy()
    out["Payout Ratio %"] = np.where(out[comp] > 0, (out[decl] / out[comp]) * 100.0, np.nan).round(2)
    out["Meets 90% Rule (SPV)"] = out["Payout Ratio %"] >= 90.0

    out["SPV+HoldCo CF Sum"] = (
        out[spv_cfo].fillna(0) + out[spv_cfi].fillna(0) + out[spv_cff].fillna(0) + out[spv_pat].fillna(0) +
        out[hco_cfo].fillna(0) + out[hco_cfi].fillna(0) + out[hco_cff].fillna(0) + out[hco_pat].fillna(0)
    )
    out["Gap vs Computed (SPV)"] = out["SPV+HoldCo CF Sum"] - out[comp]
    out["Gap % of Computed (SPV)"] = np.where(out[comp] != 0, (out["Gap vs Computed (SPV)"] / out[comp]) * 100.0, np.nan).round(2)
    # Requirement: |gap| < computed (i.e., not >= computed)
    out["Within Computed Bound (SPV)"] = np.where(out[comp] > 0, out["Gap vs Computed (SPV)"].abs() < out[comp], np.nan)
    return out

# ---- UI ---------------------------------------------------------------------
def render():
    st.header("NDCF — Compliance Checks")

    with st.sidebar:
        seg = st.selectbox("Select Segment", ["REIT", "InvIT"], index=0)
        data_url = st.text_input(
            "Data URL (Google Sheet - public view)",
            value=DEFAULT_SHEET_URL_TRUST,
            help=f"Trust sheet: '{TRUST_SHEET_NAME}'. SPV sheet: '{SPV_SHEET_NAME}'.",
        )

    if seg != "REIT":
        st.info("InvIT checks will be added later.")
        return

    df_trust_all = load_reit_ndcf(data_url, TRUST_SHEET_NAME)
    df_spv_all   = load_reit_spv_ndcf(data_url, SPV_SHEET_NAME)
    if df_trust_all.empty:
        return

    ent = st.selectbox(
        "Choose REIT",
        sorted(df_trust_all["Name of REIT"].dropna().unique().tolist()),
        index=0,
        key="ndcf_reit_select",
    )
    fy_options = sorted(df_trust_all.loc[df_trust_all["Name of REIT"] == ent, "Financial Year"].dropna().unique().tolist())
    fy = st.selectbox(
        "Financial Year",
        ["— Select —"] + fy_options,
        index=0,
        key="ndcf_fy_select",
        help="Pick a Financial Year to run the checks.",
    )
    if fy == "— Select —":
        st.info("Pick a Financial Year to show results.")
        return

    # ---------- TRUST-LEVEL ----------
    q_trust = df_trust_all[(df_trust_all["Name of REIT"] == ent) & (df_trust_all["Financial Year"] == fy)].copy()
    if q_trust.empty:
        st.warning("No TRUST-level rows for the selected REIT and Financial Year.")
    else:
        q_trust = compute_trust_checks(q_trust)

        total = int(len(q_trust))
        good_payout = int(q_trust["Meets 90% Rule"].astype("boolean").fillna(False).sum())
        good_gap = int(q_trust["Within 10% Gap"].astype("boolean").fillna(False).sum())
        s1, s2, s3 = st.columns(3)
        s1.metric("TRUST: periods meeting 90% payout", f"{good_payout}/{total}")
        s2.metric("TRUST: periods within 10% gap", f"{good_gap}/{total}")
        s3.metric("TRUST: rows analysed", f"{total}")

        st.subheader("Trust Check 1 — 90% payout of Computed NDCF (period-wise)")
        disp1 = q_trust[[
            "Financial Year",
            "Period Ended",
            "Total Amount of NDCF computed as per NDCF Statement",
            "Total Amount of NDCF declared for the period (incl. Surplus)",
            "Payout Ratio %",
            "Meets 90% Rule",
        ]].copy()
        disp1["Meets 90% Rule"] = disp1["Meets 90% Rule"].map(_status)
        st.dataframe(disp1, use_container_width=True, hide_index=True)
        if _fails(q_trust["Meets 90% Rule"]).any():
            st.error("TRUST: One or more periods do **not** meet the 90% payout requirement (Declared incl. surplus < 90% of Computed NDCF).")

        st.subheader("Trust Check 2 — (CFO + CFI + CFF + PAT) gap vs Computed NDCF (period-wise)")
        disp2 = q_trust[[
            "Financial Year",
            "Period Ended",
            "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
            "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
            "Cash Flow From Fincing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
            "Profit after tax as per Statement of Profit and Loss (as per Audited Fincials or Fincials with Limited Review)",
            "CF Sum",
            "Total Amount of NDCF computed as per NDCF Statement",
            "Gap vs Computed",
            "Gap % of Computed",
            "Within 10% Gap",
        ]].copy()
        disp2["Within 10% Gap"] = disp2["Within 10% Gap"].map(_status)
        st.dataframe(disp2, use_container_width=True, hide_index=True)
        if _fails(q_trust["Within 10% Gap"]).any():
            st.error("TRUST: One or more periods have a gap **> 10%** between (CFO + CFI + CFF + PAT) and Computed NDCF.")

    st.divider()

    # ---------- SPV-LEVEL ----------
    st.subheader("SPV/HoldCo Checks (for selected REIT + FY)")
    if df_spv_all.empty:
        st.info("SPV sheet could not be loaded or columns are missing; skipping SPV checks.")
        return

    q_spv = df_spv_all[(df_spv_all["Name of REIT"] == ent) & (df_spv_all["Financial Year"] == fy)].copy()
    if q_spv.empty:
        st.warning("No SPV-level rows for the selected REIT and Financial Year.")
        return

    q_spv = compute_spv_checks(q_spv)

    st.markdown("**SPV Check 1 — Declared (incl. Surplus) ≥ 90% of Computed NDCF (by SPV/period)**")
    disp_s1 = q_spv[[
        "Name of SPV",
        "Name of Holdco (Leave Blank if N/A)",
        "Financial Year",
        "Period Ended",
        "Total Amount of NDCF computed as per NDCF Statement",
        "Total Amount of NDCF declared for the period (incl. Surplus)",
        "Payout Ratio %",
        "Meets 90% Rule (SPV)",
    ]].copy()
    disp_s1["Meets 90% Rule (SPV)"] = disp_s1["Meets 90% Rule (SPV)"].map(_status)
    st.dataframe(disp_s1, use_container_width=True, hide_index=True)
    if _fails(q_spv["Meets 90% Rule (SPV)"]).any():
        st.error("SPV: One or more SPV periods do **not** meet the 90% payout requirement.")

    st.markdown("**SPV Check 2 — |(SPV+HoldCo CFO+CFI+CFF+PAT) − Computed NDCF| < Computed NDCF**")
    disp_s2 = q_spv[[
        "Name of SPV",
        "Name of Holdco (Leave Blank if N/A)",
        "Financial Year",
        "Period Ended",
        "SPV+HoldCo CF Sum",
        "Total Amount of NDCF computed as per NDCF Statement",
        "Gap vs Computed (SPV)",
        "Gap % of Computed (SPV)",
        "Within Computed Bound (SPV)",
    ]].copy()
    disp_s2["Within Computed Bound (SPV)"] = disp_s2["Within Computed Bound (SPV)"].map(_status)
    st.dataframe(disp_s2, use_container_width=True, hide_index=True)
    if _fails(q_spv["Within Computed Bound (SPV)"]).any():
        st.error("SPV: One or more SPV periods have |Gap| ≥ Computed NDCF.")

# Exported entrypoints for pages/5_NDCF.py
def render_ndcf():
    render()

if __name__ == "__main__":
    render()
