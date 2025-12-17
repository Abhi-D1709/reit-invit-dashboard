# tabs/ndcf.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Optional

# ---- Config (overridden by utils.common if present) -------------------------
DEFAULT_SHEET_URL_TRUST = "https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing"
TRUST_SHEET_NAME = "NDCF REITs"
SPV_SHEET_NAME   = "NDCF SPV REIT"

DEFAULT_REIT_DIR_URL = None  # Offer Document links live here (Sheet5)

try:
    # Centralized constants (if present)
    from utils.common import NDCF_REITS_SHEET_URL, DEFAULT_REIT_DIR_URL as _DIR_URL
    if NDCF_REITS_SHEET_URL:
        DEFAULT_SHEET_URL_TRUST = NDCF_REITS_SHEET_URL
    if _DIR_URL:
        DEFAULT_REIT_DIR_URL = _DIR_URL
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

def _status(v: Optional[bool]) -> str:
    if pd.isna(v):
        return "—"
    return "🟢" if bool(v) else "🔴"

def _to_date(s) -> pd.Timestamp:
    """Parse to Timestamp (date-only). Returns NaT if not parseable."""
    if pd.isna(s):
        return pd.NaT
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return pd.NaT
        return pd.to_datetime(dt.date())
    except Exception:
        return pd.NaT

# ---- TRUST-LEVEL -------------------------------------------------------------
def load_reit_ndcf(url: str, sheet_name: str = TRUST_SHEET_NAME) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Entity": "Name of REIT",
        "Financial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Date of Finalisation/Declaration of NDCF Statement by REIT": "Declaration Date",
        "Record Date": "Record Date",
        "Date of Distribution of NDCF by REIT": "Distribution Date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    def _find_col(cols, *tokens):
        for c in cols:
            cl = c.lower().replace("finalisation", "finalization")
            if all(t in cl for t in tokens):
                return c
        return None

    if "Declaration Date" not in df.columns:
        cand = _find_col(df.columns, "declar") or _find_col(df.columns, "finaliz", "ndcf")
        if cand:
            df = df.rename(columns={cand: "Declaration Date"})
    if "Record Date" not in df.columns:
        cand = _find_col(df.columns, "record", "date")
        if cand:
            df = df.rename(columns={cand: "Record Date"})
    if "Distribution Date" not in df.columns:
        cand = _find_col(df.columns, "distribution", "date")
        if cand:
            df = df.rename(columns={cand: "Distribution Date"})

    needed = [
        "Name of REIT",
        "Financial Year",
        "Period Ended",
        "Total Amount of NDCF computed as per NDCF Statement",
        "Total Amount of NDCF declared for the period (incl. Surplus)",
        "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)",
        "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)",
        "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)",
        "Profit after tax as per Statement of Profit and Loss (as per Audited Financials with Limited Review)",
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

    if "Declaration Date" in df.columns:
        df["Declaration Date"] = df["Declaration Date"].map(_to_date)
    if "Record Date" in df.columns:
        df["Record Date"] = df["Record Date"].map(_to_date)
    if "Distribution Date" in df.columns:
        df["Distribution Date"] = df["Distribution Date"].map(_to_date)

    return df

def compute_trust_checks(df: pd.DataFrame) -> pd.DataFrame:
    comp = "Total Amount of NDCF computed as per NDCF Statement"
    decl = "Total Amount of NDCF declared for the period (incl. Surplus)"
    cfo = "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)"
    cfi = "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)"
    cff = "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)"
    pat = "Profit after tax as per Statement of Profit and Loss (as per Audited Financials with Limited Review)"

    out = df.copy()
    # Check 1: ≥ 90% payout
    out["Payout Ratio %"] = np.where(out[comp] > 0, (out[decl] / out[comp]) * 100.0, np.nan).round(2)
    out["Meets 90% Rule"] = out["Payout Ratio %"] >= 90.0

    # Check 2: |(CFO+CFI+CFF+PAT − Computed)| ≤ 10% of Computed
    out["CF Sum"] = out[cfo].fillna(0) + out[cfi].fillna(0) + out[cff].fillna(0) + out[pat].fillna(0)
    out["Gap vs Computed"] = out["CF Sum"] - out[comp]
    out["Gap % of Computed"] = np.where(out[comp] != 0, (out["Gap vs Computed"] / out[comp]) * 100.0, np.nan).round(2)
    out["Within 10% Gap"] = out["Gap % of Computed"].abs() <= 10.0

    return out

def compute_trust_timeline_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check the date gaps
      - Declaration → Record Date <= 2 days
      - Record Date → Distribution Date <= 5 days
    """
    if not {"Declaration Date", "Record Date", "Distribution Date"}.issubset(df.columns):
        return pd.DataFrame(columns=[
            "Financial Year", "Period Ended",
            "Declaration Date", "Record Date", "Distribution Date",
            "Days Decl→Record", "Record ≤ 2 days",
            "Days Record→Distr", "Distribution ≤ 5 days"
        ])

    t = df.copy()
    t["Days Decl→Record"] = (t["Record Date"] - t["Declaration Date"]).dt.days
    t["Days Record→Distr"] = (t["Distribution Date"] - t["Record Date"]).dt.days

    t["Record ≤ 2 days"] = (t["Days Decl→Record"] >= 0) & (t["Days Decl→Record"] <= 2)
    t["Distribution ≤ 5 days"] = (t["Days Record→Distr"] >= 0) & (t["Days Record→Distr"] <= 5)

    return t[[
        "Financial Year", "Period Ended",
        "Declaration Date", "Record Date", "Distribution Date",
        "Days Decl→Record", "Record ≤ 2 days",
        "Days Record→Distr", "Distribution ≤ 5 days"
    ]].copy()

# ---- SPV-LEVEL ---------------------------------------------------------------
def load_reit_spv_ndcf(url: str, sheet_name: str = SPV_SHEET_NAME) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Entity": "Name of REIT",
        "Financial Year": "Financial Year",
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
    out["Within Computed Bound (SPV)"] = np.where(out[comp] > 0, out["Gap vs Computed (SPV)"].abs() < out[comp], np.nan)
    return out

# ---- Offer Document (Sheet5 of Basic Details) --------------------------------
def load_offer_doc_links(dir_url: Optional[str]) -> pd.DataFrame:
    """
    Loads 'Sheet5' of the Basic Details workbook and returns a map:
      Name of REIT -> OD Link
    """
    if not dir_url:
        return pd.DataFrame(columns=["Name of REIT", "OD Link"])
    try:
        csv_url = _csv_url_from_gsheet(dir_url, sheet="Sheet5")
        df = pd.read_csv(csv_url)
        df.columns = [c.strip() for c in df.columns]
        ent_col = next((c for c in df.columns if "name" in c.lower() and "reit" in c.lower()), None)
        link_col = next((c for c in df.columns if "od" in c.lower() and "link" in c.lower()), None)
        if not ent_col:
            ent_col = "Name of REIT" if "Name of REIT" in df.columns else df.columns[0]
        if not link_col:
            candidates = [c for c in df.columns if "link" in c.lower() or "http" in "".join(df[c].astype(str).tolist()).lower()]
            link_col = candidates[0] if candidates else df.columns[-1]
        return df[[ent_col, link_col]].rename(columns={ent_col: "Name of REIT", link_col: "OD Link"})
    except Exception:
        return pd.DataFrame(columns=["Name of REIT", "OD Link"])

# ---- UI (flow: REIT -> Level -> FY) -----------------------------------------
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

    # Load both sheets once
    df_trust_all = load_reit_ndcf(data_url, TRUST_SHEET_NAME)
    df_spv_all   = load_reit_spv_ndcf(data_url, SPV_SHEET_NAME)
    if df_trust_all.empty:
        return

    # 1) Choose REIT first
    ent = st.selectbox(
        "Choose REIT",
        sorted(df_trust_all["Name of REIT"].dropna().unique().tolist()),
        index=0,
        key="ndcf_reit_select",
    )

    # Offer Document link (from Basic Details / Sheet5)
    if DEFAULT_REIT_DIR_URL:
        od_df = load_offer_doc_links(DEFAULT_REIT_DIR_URL)
        od_match = od_df.loc[od_df["Name of REIT"] == ent, "OD Link"]
        if not od_match.empty and isinstance(od_match.iloc[0], str) and od_match.iloc[0].strip():
            st.markdown(f"**Offer Document:** [{od_match.iloc[0].strip()}]({od_match.iloc[0].strip()})")

    # 2) Choose analysis level next (Trust vs SPV)
    level = st.radio("Analysis level", ["Trust", "SPV/HoldCo"], horizontal=True, key="ndcf_level_select")

    # 3) Choose FY (options depend on the chosen level)
    if level == "Trust":
        fy_options = sorted(df_trust_all.loc[df_trust_all["Name of REIT"] == ent, "Financial Year"].dropna().unique().tolist())
    else:
        fy_options = [] if df_spv_all.empty else sorted(
            df_spv_all.loc[df_spv_all["Name of REIT"] == ent, "Financial Year"].dropna().unique().tolist()
        )

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

    # ---------- TRUST-LEVEL (if selected) ----------
    if level == "Trust":
        q_trust = df_trust_all[(df_trust_all["Name of REIT"] == ent) & (df_trust_all["Financial Year"] == fy)].copy()
        if q_trust.empty:
            st.warning("No TRUST-level rows for the selected REIT and Financial Year.")
            return

        # Core checks (90% payout / 10% gap)
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
        if (~q_trust["Meets 90% Rule"].astype("boolean").fillna(False)).any():
            st.error("TRUST: One or more periods do **not** meet the 90% payout requirement (Declared incl. surplus < 90% of Computed NDCF).")

        st.subheader("Trust Check 2 — (CFO + CFI + CFF + PAT) gap vs Computed NDCF (period-wise)")
        disp2 = q_trust[[
            "Financial Year",
            "Period Ended",
            "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)",
            "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)",
            "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)",
            "Profit after tax as per Statement of Profit and Loss (as per Audited Financials with Limited Review)",
            "CF Sum",
            "Total Amount of NDCF computed as per NDCF Statement",
            "Gap vs Computed",
            "Gap % of Computed",
            "Within 10% Gap",
        ]].copy()
        disp2["Within 10% Gap"] = disp2["Within 10% Gap"].map(_status)
        st.dataframe(disp2, use_container_width=True, hide_index=True)
        if (~q_trust["Within 10% Gap"].astype("boolean").fillna(False)).any():
            st.error("TRUST: One or more periods have a gap **> 10%** between (CFO + CFI + CFF + PAT) and Computed NDCF.")

        # -------- NEW: timeline checks split into two separate tables --------
        tline = compute_trust_timeline_checks(q_trust)
        if tline.empty:
            st.info("Declaration / Record / Distribution columns not found; timeline checks skipped.")
        else:
            # 3a) Declaration → Record Date (≤ 2 days)
            st.subheader("Trust Check 3a — Declaration → Record Date (≤ 2 days)")
            t1 = tline[[
                "Financial Year", "Period Ended",
                "Declaration Date", "Record Date",
                "Days Decl→Record", "Record ≤ 2 days"
            ]].copy()
            t1["Record ≤ 2 days"] = t1["Record ≤ 2 days"].map(_status)
            st.dataframe(t1, use_container_width=True, hide_index=True)

            bad1 = (tline["Record ≤ 2 days"] == False)
            if bad1.any():
                st.error("TRUST: One or more periods have **Record Date more than 2 days after Declaration**.")

            # 3b) Record Date → Distribution Date (≤ 5 days)
            st.subheader("Trust Check 3b — Record Date → Distribution Date (≤ 5 days)")
            t2 = tline[[
                "Financial Year", "Period Ended",
                "Record Date", "Distribution Date",
                "Days Record→Distr", "Distribution ≤ 5 days"
            ]].copy()
            t2["Distribution ≤ 5 days"] = t2["Distribution ≤ 5 days"].map(_status)
            st.dataframe(t2, use_container_width=True, hide_index=True)

            bad2 = (tline["Distribution ≤ 5 days"] == False)
            if bad2.any():
                st.error("TRUST: One or more periods have **Distribution Date more than 5 days after Record Date**.")

    # ---------- SPV-LEVEL (if selected) ----------
    else:
        if df_spv_all.empty:
            st.info("SPV sheet could not be loaded or columns are missing; skipping SPV checks.")
            return

        q_spv = df_spv_all[(df_spv_all["Name of REIT"] == ent) & (df_spv_all["Financial Year"] == fy)].copy()
        if q_spv.empty:
            st.warning("No SPV-level rows for the selected REIT and Financial Year.")
            return

        q_spv = compute_spv_checks(q_spv)

        st.subheader("SPV Check 1 — Declared (incl. Surplus) ≥ 90% of Computed NDCF (by SPV/period)")
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
        if (~q_spv["Meets 90% Rule (SPV)"].astype("boolean").fillna(False)).any():
            st.error("SPV: One or more SPV periods do **not** meet the 90% payout requirement.")

        st.subheader("SPV Check 2 — |(SPV+HoldCo CFO+CFI+CFF+PAT) − Computed NDCF| < Computed NDCF")
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
        if (~q_spv["Within Computed Bound (SPV)"].astype("boolean").fillna(False)).any():
            st.error("SPV: One or more SPV periods have |Gap| ≥ Computed NDCF.")

# Exported entrypoint for pages/5_NDCF.py
def render_ndcf():
    render()

if __name__ == "__main__":
    render()
