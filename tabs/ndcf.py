# tabs/ndcf.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Optional

# -------- Defaults --------
DEFAULT_SHEET_URL_TRUST = "https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing"
TRUST_SHEET_NAME = "NDCF REITs"
SPV_SHEET_NAME   = "NDCF SPV REIT"
DEFAULT_REIT_DIR_URL = None

try:
    from utils.common import NDCF_REITS_SHEET_URL, DEFAULT_REIT_DIR_URL as _DIR_URL
    if NDCF_REITS_SHEET_URL:
        DEFAULT_SHEET_URL_TRUST = NDCF_REITS_SHEET_URL
    if _DIR_URL:
        DEFAULT_REIT_DIR_URL = _DIR_URL
except ImportError:
    pass

# --- CONSTANTS (Updated to Standard "Financials" and "Financing") ---

# 1. TRUST LEVEL
COMP_COL = "Total Amount of NDCF computed as per NDCF Statement"
DECL_INCL_COL = "Total Amount of NDCF declared for the period (incl. Surplus)"

# UPDATED: Assumes "Fincing" was also corrected to "Financing" in your sheet
CFO_COL = "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
CFI_COL = "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
CFF_COL = "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
PAT_COL = "Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"

# 2. SPV LEVEL
SPV_CFO = "SPV Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
SPV_CFI = "SPV Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
SPV_CFF = "SPV Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
SPV_PAT = "SPV Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"
HCO_CFO = "HoldCo Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
HCO_CFI = "HoldCo Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
HCO_CFF = "Holdco Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
HCO_PAT = "Holdco Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"

# ------------------------ helpers ------------------------
def _csv_url_from_gsheet(url: str, sheet: Optional[str] = None, gid: Optional[str] = None) -> str:
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        return url
    sheet_id = m.group(1)
    
    if not gid:
        m_gid = re.search(r"[?&]gid=(\d+)", url)
        if m_gid:
            gid = m_gid.group(1)
    
    if gid:
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    
    if sheet:
        from urllib.parse import quote
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet)}"
    
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

def _clean_numeric_col(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(",", "")
    s = s.replace({"-": np.nan, "–": np.nan, "—": np.nan, "": np.nan, "None": np.nan, "nan": np.nan})
    mask_parens = s.str.match(r"^\(.*\)$", na=False)
    s.loc[mask_parens] = "-" + s.loc[mask_parens].str[1:-1]
    return pd.to_numeric(s, errors='coerce')

def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    s_str = series.astype(str).str.strip()
    s_str = s_str.replace({"None": np.nan, "nan": np.nan, "": np.nan})
    dates = pd.to_datetime(s_str, dayfirst=True, errors='coerce')
    mask_nat = dates.isna() & s_str.notna()
    if mask_nat.any():
        numeric_part = pd.to_numeric(s_str[mask_nat], errors='coerce')
        valid_serials = numeric_part[(numeric_part > 10000) & (numeric_part < 80000)]
        if not valid_serials.empty:
            dates_serial = pd.to_datetime(valid_serials, unit='D', origin='1899-12-30')
            dates.update(dates_serial)
    return dates

def _status(v) -> str:
    if pd.isna(v) or v is None:
        return "—"
    return "🟢" if bool(v) else "🔴"

# ------------------- loaders -------------------

@st.cache_data(ttl=600)
def load_reit_ndcf(url: str, sheet_name: str = TRUST_SHEET_NAME) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
    try:
        df = pd.read_csv(csv_url, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame()
        
    # Robust header cleaning
    df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]

    rename_map = {
        "Entity": "Name of REIT",
        "Fincial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Date of Finalisation/Declaration of NDCF Statement by REIT": "Declaration Date",
        "Record Date": "Record Date",
        "Date of Distribution of NDCF by REIT": "Distribution Date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["Declaration Date", "Record Date", "Distribution Date"]:
        if col in df.columns:
            df[f"{col}__raw"] = df[col]
            df[col] = _parse_mixed_dates(df[col])

    # Try to clean numeric columns if they exist
    for c in [COMP_COL, DECL_INCL_COL, CFO_COL, CFI_COL, CFF_COL, PAT_COL]:
        if c in df.columns:
            df[c] = _clean_numeric_col(df[c])

    for c in ["Name of REIT", "Financial Year", "Period Ended"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

@st.cache_data(ttl=600)
def load_reit_spv_ndcf(url: str, sheet_name: str = SPV_SHEET_NAME) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
    try:
        df = pd.read_csv(csv_url, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame()
        
    df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]

    rename_map = {
        "Entity": "Name of REIT",
        "Fincial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Name of Holdco": "Name of Holdco (Leave Blank if N/A)",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    numeric_cols = [
        COMP_COL, DECL_INCL_COL, 
        SPV_CFO, SPV_CFI, SPV_CFF, SPV_PAT, 
        HCO_CFO, HCO_CFI, HCO_CFF, HCO_PAT
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = _clean_numeric_col(df[c])

    for c in ["Name of REIT","Financial Year","Period Ended","Name of SPV","Name of Holdco (Leave Blank if N/A)"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

@st.cache_data(ttl=3600)
def load_offer_doc_links(dir_url: Optional[str]) -> pd.DataFrame:
    if not dir_url:
        return pd.DataFrame(columns=["Name of REIT","OD Link"])
    try:
        csv_url = _csv_url_from_gsheet(dir_url, sheet="Sheet5")
        df = pd.read_csv(csv_url, dtype=str, keep_default_na=False)
        df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]
        
        ent_col = next((c for c in df.columns if "name" in c.lower() and "reit" in c.lower()), None)
        link_col = next((c for c in df.columns if "link" in c.lower()), None)
        
        if ent_col and link_col:
            return df[[ent_col, link_col]].rename(columns={ent_col: "Name of REIT", link_col: "OD Link"})
        return pd.DataFrame(columns=["Name of REIT","OD Link"])
    except Exception:
        return pd.DataFrame(columns=["Name of REIT","OD Link"])

# ------------------- computations -------------------
def compute_trust_checks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # SAFETY: Check if columns exist before calculation
    missing_cols = []
    for col in [CFO_COL, CFI_COL, CFF_COL, PAT_COL]:
        if col not in out.columns:
            missing_cols.append(col)
    
    if missing_cols:
        st.error(f"❌ Missing Columns in Data: {missing_cols}")
        st.write("Please check your Google Sheet headers exactly match what the code expects.")
        return pd.DataFrame() # Return empty to prevent crash

    out["Payout Ratio %"] = np.nan
    mask_nonzero = out[COMP_COL] != 0
    out.loc[mask_nonzero, "Payout Ratio %"] = (out.loc[mask_nonzero, DECL_INCL_COL] / out.loc[mask_nonzero, COMP_COL]) * 100.0
    out["Payout Ratio %"] = out["Payout Ratio %"].round(2)
    out["Meets 90% Rule"] = out["Payout Ratio %"] >= 90.0

    out["CF Sum"] = out[CFO_COL].fillna(0) + out[CFI_COL].fillna(0) + out[CFF_COL].fillna(0) + out[PAT_COL].fillna(0)
    out["Gap vs Computed"] = out["CF Sum"] - out[COMP_COL]
    
    out["Gap % of Computed"] = np.nan
    out.loc[mask_nonzero, "Gap % of Computed"] = (out.loc[mask_nonzero, "Gap vs Computed"] / out.loc[mask_nonzero, COMP_COL]) * 100.0
    out["Gap % of Computed"] = out["Gap % of Computed"].round(2)
    out["Within 10% Gap"] = out["Gap % of Computed"].abs() <= 10.0
    return out

def compute_trust_timeline_checks(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"Declaration Date","Record Date","Distribution Date"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    
    t = df.copy()
    t["Days Decl→Record"] = (t["Record Date"] - t["Declaration Date"]).dt.days
    t["Days Record→Distr"] = (t["Distribution Date"] - t["Record Date"]).dt.days
    t["Record ≤ 2 days"] = (t["Days Decl→Record"] >= 0) & (t["Days Decl→Record"] <= 2)
    t["Distribution ≤ 5 days"] = (t["Days Record→Distr"] >= 0) & (t["Days Record→Distr"] <= 5)
    
    return t[[
        "Financial Year","Period Ended","Declaration Date","Record Date","Distribution Date",
        "Days Decl→Record","Record ≤ 2 days","Days Record→Distr","Distribution ≤ 5 days"
    ]].copy()

def compute_spv_checks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Payout Ratio %"] = np.nan
    mask_nonzero = out[COMP_COL] != 0
    out.loc[mask_nonzero, "Payout Ratio %"] = (out.loc[mask_nonzero, DECL_INCL_COL] / out.loc[mask_nonzero, COMP_COL]) * 100.0
    out["Payout Ratio %"] = out["Payout Ratio %"].round(2)
    out["Meets 90% Rule (SPV)"] = out["Payout Ratio %"] >= 90.0

    out["SPV+HoldCo CF Sum"] = (
        out[SPV_CFO].fillna(0) + out[SPV_CFI].fillna(0) + out[SPV_CFF].fillna(0) + out[SPV_PAT].fillna(0) +
        out[HCO_CFO].fillna(0) + out[HCO_CFI].fillna(0) + out[HCO_CFF].fillna(0) + out[HCO_PAT].fillna(0)
    )
    out["Gap vs Computed (SPV)"] = out["SPV+HoldCo CF Sum"] - out[COMP_COL]
    
    out["Gap % of Computed (SPV)"] = np.nan
    out.loc[mask_nonzero, "Gap % of Computed (SPV)"] = (out.loc[mask_nonzero, "Gap vs Computed (SPV)"] / out.loc[mask_nonzero, COMP_COL]) * 100.0
    out["Gap % of Computed (SPV)"] = out["Gap % of Computed (SPV)"].round(2)
    
    out["Within Computed Bound (SPV)"] = np.where(
        out[COMP_COL] > 0, 
        out["Gap vs Computed (SPV)"].abs() < out[COMP_COL], 
        np.nan
    )
    return out

# ------------------- UI -------------------
def render():
    st.header("NDCF — Compliance Checks")

    with st.sidebar:
        seg = st.selectbox("Select Segment", ["REIT", "InvIT"], index=0)
        data_url = st.text_input(
            "Data URL (Google Sheet)",
            value=DEFAULT_SHEET_URL_TRUST,
            help="Full URL including 'gid=' is recommended for best reliability.",
        )

    if seg != "REIT":
        st.info("InvIT checks will be added later.")
        return

    df_trust_all = load_reit_ndcf(data_url, TRUST_SHEET_NAME)
    df_spv_all   = load_reit_spv_ndcf(data_url, SPV_SHEET_NAME)

    # --- DEBUG PANEL ---
    with st.expander("🛠️ Debug: CSV Data Inspector", expanded=False):
        st.write("Columns Detected (Trust):", df_trust_all.columns.tolist())
        st.write("First 3 Rows (Trust):")
        st.dataframe(df_trust_all.head(3), use_container_width=True)

    if df_trust_all.empty:
        st.warning("Trust sheet appears empty or columns not found.")
        return

    # Selectors
    reit_list = sorted(df_trust_all["Name of REIT"].dropna().unique().tolist())
    ent = st.selectbox("Choose REIT", reit_list, index=0, key="ndcf_reit_select")

    if DEFAULT_REIT_DIR_URL:
        od_df = load_offer_doc_links(DEFAULT_REIT_DIR_URL)
        od = od_df.loc[od_df["Name of REIT"] == ent, "OD Link"]
        if not od.empty and pd.notna(od.iloc[0]) and str(od.iloc[0]).strip() != "":
            st.markdown(f"📄 **Offer Document:** [{od.iloc[0].strip()}]({od.iloc[0].strip()})")

    fy_options = sorted(df_trust_all.loc[df_trust_all["Name of REIT"] == ent, "Financial Year"].dropna().unique().tolist())
    fy = st.selectbox("Financial Year", ["— Select —"] + fy_options, index=0, key="ndcf_fy_select")

    if fy == "— Select —":
        st.info("Pick a Financial Year to show results.")
        return

    tab_trust, tab_spv = st.tabs(["🏛️ Trust Level", "🏗️ SPV/HoldCo Level"])

    with tab_trust:
        q = df_trust_all[(df_trust_all["Name of REIT"] == ent) & (df_trust_all["Financial Year"] == fy)].copy()
        
        if q.empty:
            st.warning("No data for this selection.")
        else:
            qc = compute_trust_checks(q)
            if not qc.empty:
                c1, c2, c3 = st.columns(3)
                fails_90 = (~qc["Meets 90% Rule"].astype("boolean").fillna(False)).sum()
                fails_gap = (~qc["Within 10% Gap"].astype("boolean").fillna(False)).sum()
                c1.metric("90% Rule Failures", f"{fails_90}", delta_color="inverse")
                c2.metric("Gap Check Failures", f"{fails_gap}", delta_color="inverse")
                c3.metric("Records Analyzed", f"{len(qc)}")
                st.divider()

                st.subheader("1. 90% Payout Rule")
                disp1 = qc[[
                    "Financial Year","Period Ended",
                    COMP_COL, DECL_INCL_COL,
                    "Payout Ratio %","Meets 90% Rule",
                ]].copy()
                disp1["Meets 90% Rule"] = disp1["Meets 90% Rule"].map(_status)
                st.dataframe(disp1, use_container_width=True, hide_index=True)

                st.subheader("2. Cash Flow Reconciliation")
                disp2 = qc[[
                    "Period Ended", "CF Sum", COMP_COL, "Gap vs Computed", "Gap % of Computed", "Within 10% Gap"
                ]].copy()
                disp2 = disp2.rename(columns={COMP_COL: "Computed NDCF"})
                disp2["Within 10% Gap"] = disp2["Within 10% Gap"].map(_status)
                st.dataframe(disp2, use_container_width=True, hide_index=True)

                st.subheader("3. Timeline Compliance")
                tline = compute_trust_timeline_checks(q)
                if not tline.empty:
                    t1 = tline[[
                        "Period Ended","Declaration Date","Record Date","Days Decl→Record","Record ≤ 2 days"
                    ]].copy()
                    t1["Record ≤ 2 days"] = t1["Record ≤ 2 days"].map(_status)
                    
                    t2 = tline[[
                        "Period Ended","Record Date","Distribution Date","Days Record→Distr","Distribution ≤ 5 days"
                    ]].copy()
                    t2["Distribution ≤ 5 days"] = t2["Distribution ≤ 5 days"].map(_status)

                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.write("**Decl → Record (≤2 days)**")
                        st.dataframe(t1, use_container_width=True, hide_index=True)
                    with col_t2:
                        st.write("**Record → Distr (≤5 days)**")
                        st.dataframe(t2, use_container_width=True, hide_index=True)
                else:
                    st.warning("Date columns missing (check Debug section).")

                with st.expander("Debug: Raw Date Strings"):
                    cols = [c for c in ["Declaration Date__raw","Record Date__raw","Distribution Date__raw"] if c in q.columns]
                    st.dataframe(q[cols], use_container_width=True, hide_index=True)

    with tab_spv:
        if df_spv_all.empty:
            st.info("SPV data not loaded.")
        else:
            q_spv = df_spv_all[(df_spv_all["Name of REIT"] == ent) & (df_spv_all["Financial Year"] == fy)].copy()
            if q_spv.empty:
                st.warning("No SPV data for this selection.")
            else:
                qs = compute_spv_checks(q_spv)
                st.subheader("1. SPV 90% Payout")
                d1 = qs[[
                    "Name of SPV","Period Ended",
                    COMP_COL, DECL_INCL_COL, "Payout Ratio %","Meets 90% Rule (SPV)"
                ]].copy()
                d1 = d1.rename(columns={COMP_COL:"Computed", DECL_INCL_COL:"Declared"})
                d1["Meets 90% Rule (SPV)"] = d1["Meets 90% Rule (SPV)"].map(_status)
                st.dataframe(d1, use_container_width=True, hide_index=True)

                st.subheader("2. SPV + HoldCo Reconciliation")
                d2 = qs[[
                    "Name of SPV","Period Ended",
                    "SPV+HoldCo CF Sum", COMP_COL, "Gap vs Computed (SPV)","Within Computed Bound (SPV)"
                ]].copy()
                d2 = d2.rename(columns={COMP_COL:"Computed NDCF"})
                d2["Within Computed Bound (SPV)"] = d2["Within Computed Bound (SPV)"].map(_status)
                st.dataframe(d2, use_container_width=True, hide_index=True)

def render_ndcf():
    render()

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render()