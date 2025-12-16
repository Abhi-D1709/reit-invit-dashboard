# tabs/ndcf.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Optional

# ---------- Config (overridden by utils.common if present) ----------
DEFAULT_SHEET_URL_TRUST = "https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing"
TRUST_SHEET_NAME = "NDCF REITs"
SPV_SHEET_NAME   = "NDCF SPV REIT"
DEFAULT_REIT_DIR_URL = None  # workbook that holds OD links in "Sheet5"

try:
    from utils.common import NDCF_REITS_SHEET_URL, DEFAULT_REIT_DIR_URL as _DIR_URL
    if NDCF_REITS_SHEET_URL:
        DEFAULT_SHEET_URL_TRUST = NDCF_REITS_SHEET_URL
    if _DIR_URL:
        DEFAULT_REIT_DIR_URL = _DIR_URL
except Exception:
    pass

# ---------- helpers ----------
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

def _strip(s):
    return str(s).strip() if pd.notna(s) else s

def _to_number(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s in {"-", "–", "—"}:
        return np.nan
    s = s.replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except Exception:
        return np.nan

def _status(v: Optional[bool]) -> str:
    if pd.isna(v):
        return "—"
    return "🟢" if bool(v) else "🔴"

# --------- robust date parsing from RAW text ----------
def _to_date(val) -> pd.Timestamp:
    if val is None:
        return pd.NaT
    # Excel serial passed as number-like string
    s = str(val).strip()
    if s == "" or s.lower() in {"none", "null", "na", "-", "nat"}:
        return pd.NaT

    # numeric serials
    if re.fullmatch(r"\d{5,6}(\.\d+)?", s):
        try:
            f = float(s)
            if 10000 <= f <= 80000:
                return pd.to_datetime(f, unit="D", origin="1899-12-30", errors="coerce")
        except Exception:
            pass

    # first try flexible day-first
    try:
        dt = pd.to_datetime(s, errors="raise", dayfirst=True, infer_datetime_format=True)
        return pd.to_datetime(dt.date())
    except Exception:
        pass

    # explicit formats
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = pd.to_datetime(s, format=fmt, errors="raise", dayfirst=True)
            return pd.to_datetime(dt.date())
        except Exception:
            continue

    # sanitize stray chars (NBSP, zero-width, commas, etc.)
    s2 = re.sub(r"[^0-9/\-]", "", s)
    try:
        dt = pd.to_datetime(s2, errors="raise", dayfirst=True, infer_datetime_format=True)
        return pd.to_datetime(dt.date())
    except Exception:
        return pd.NaT

# ---------- TRUST-level load ----------
def load_reit_ndcf(url: str, sheet_name: str = TRUST_SHEET_NAME) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
    # IMPORTANT: read raw text exactly as-is
    df = pd.read_csv(csv_url, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    # normalize key headers
    rename_map = {
        "Entity": "Name of REIT",
        "Fincial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Date of Finalisation/Declaration of NDCF Statement by REIT": "Declaration Date",
        "Date of Filisation/Declaration of NDCF Statement by REIT": "Declaration Date",
        "Record Date": "Record Date",
        "Date of Distribution of NDCF by REIT": "Distribution Date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # keep RAW copies of date columns
    for col in ["Declaration Date", "Record Date", "Distribution Date"]:
        if col in df.columns:
            df[f"{col}__raw"] = df[col]

    # numeric conversions (safe, starting from strings)
    numeric_cols = [
        "Total Amount of NDCF computed as per NDCF Statement",
        "Total Amount of NDCF declared for the period (incl. Surplus)",
        "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
        "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
        "Cash Flow From Fincing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)",
        "Profit after tax as per Statement of Profit and Loss (as per Audited Fincials or Fincials with Limited Review)",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].map(_to_number)

    for c in ["Name of REIT", "Financial Year", "Period Ended"]:
        if c in df.columns:
            df[c] = df[c].map(_strip)

    # parse dates from RAW
    if "Declaration Date__raw" in df.columns:
        df["Declaration Date"] = df["Declaration Date__raw"].map(_to_date)
    if "Record Date__raw" in df.columns:
        df["Record Date"] = df["Record Date__raw"].map(_to_date)
    if "Distribution Date__raw" in df.columns:
        df["Distribution Date"] = df["Distribution Date__raw"].map(_to_date)

    return df

# ---------- TRUST checks ----------
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

def compute_trust_timeline_checks(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Declaration Date","Record Date","Distribution Date"}.issubset(df.columns):
        return pd.DataFrame(columns=[
            "Financial Year","Period Ended","Declaration Date","Record Date","Distribution Date",
            "Days Decl→Record","Record ≤ 2 days","Days Record→Distr","Distribution ≤ 5 days"
        ])
    t = df.copy()
    t["Days Decl→Record"] = (t["Record Date"] - t["Declaration Date"]).dt.days
    t["Days Record→Distr"] = (t["Distribution Date"] - t["Record Date"]).dt.days
    t["Record ≤ 2 days"] = (t["Days Decl→Record"] >= 0) & (t["Days Decl→Record"] <= 2)
    t["Distribution ≤ 5 days"] = (t["Days Record→Distr"] >= 0) & (t["Days Record→Distr"] <= 5)
    return t[[
        "Financial Year","Period Ended","Declaration Date","Record Date","Distribution Date",
        "Days Decl→Record","Record ≤ 2 days","Days Record→Distr","Distribution ≤ 5 days"
    ]].copy()

# ---------- SPV-level ----------
def load_reit_spv_ndcf(url: str, sheet_name: str = SPV_SHEET_NAME) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
    df = pd.read_csv(csv_url, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Entity": "Name of REIT",
        "Fincial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Name of Holdco": "Name of Holdco (Leave Blank if N/A)",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    needed_nums = [
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
    for c in needed_nums:
        if c in df.columns:
            df[c] = df[c].map(_to_number)

    for c in ["Name of REIT","Financial Year","Period Ended","Name of SPV","Name of Holdco (Leave Blank if N/A)"]:
        if c in df.columns:
            df[c] = df[c].map(_strip)

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

# ---------- OD links ----------
def load_offer_doc_links(dir_url: Optional[str]) -> pd.DataFrame:
    if not dir_url:
        return pd.DataFrame(columns=["Name of REIT","OD Link"])
    try:
        csv_url = _csv_url_from_gsheet(dir_url, sheet="Sheet5")
        df = pd.read_csv(csv_url, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        ent_col = next((c for c in df.columns if "name" in c.lower() and "reit" in c.lower()), df.columns[0])
        link_col = next((c for c in df.columns if "link" in c.lower()), df.columns[-1])
        return df[[ent_col, link_col]].rename(columns={ent_col: "Name of REIT", link_col: "OD Link"})
    except Exception:
        return pd.DataFrame(columns=["Name of REIT","OD Link"])

# ---------- UI ----------
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
        st.warning("Trust sheet appears empty or columns not found.")
        return

    ent = st.selectbox(
        "Choose REIT",
        sorted(df_trust_all["Name of REIT"].dropna().unique().tolist()),
        index=0,
        key="ndcf_reit_select",
    )

    # OD link
    if DEFAULT_REIT_DIR_URL:
        od_df = load_offer_doc_links(DEFAULT_REIT_DIR_URL)
        od = od_df.loc[od_df["Name of REIT"] == ent, "OD Link"]
        if not od.empty and od.iloc[0].strip():
            st.markdown(f"**Offer Document:** [{od.iloc[0].strip()}]({od.iloc[0].strip()})")

    level = st.radio("Analysis level", ["Trust", "SPV/HoldCo"], horizontal=True, key="ndcf_level_select")

    if level == "Trust":
        fy_options = sorted(df_trust_all.loc[df_trust_all["Name of REIT"] == ent, "Financial Year"].dropna().unique().tolist())
    else:
        fy_options = [] if df_spv_all.empty else sorted(
            df_spv_all.loc[df_spv_all["Name of REIT"] == ent, "Financial Year"].dropna().unique().tolist()
        )

    fy = st.selectbox("Financial Year", ["— Select —"] + fy_options, index=0, key="ndcf_fy_select")
    if fy == "— Select —":
        st.info("Pick a Financial Year to show results.")
        return

    if level == "Trust":
        q = df_trust_all[(df_trust_all["Name of REIT"] == ent) & (df_trust_all["Financial Year"] == fy)].copy()
        if q.empty:
            st.warning("No TRUST-level rows for the selected REIT and Financial Year.")
            return

        # checks 1 & 2
        qc = compute_trust_checks(q)
        total = int(len(qc))
        s1, s2, s3 = st.columns(3)
        s1.metric("TRUST: periods meeting 90% payout", f"{int(qc['Meets 90% Rule'].astype('boolean').sum())}/{total}")
        s2.metric("TRUST: periods within 10% gap", f"{int(qc['Within 10% Gap'].astype('boolean').sum())}/{total}")
        s3.metric("TRUST: rows analysed", f"{total}")

        st.subheader("Trust Check 1 — 90% payout of Computed NDCF")
        disp1 = qc[[
            "Financial Year","Period Ended",
            "Total Amount of NDCF computed as per NDCF Statement",
            "Total Amount of NDCF declared for the period (incl. Surplus)",
            "Payout Ratio %","Meets 90% Rule",
        ]].copy()
        disp1["Meets 90% Rule"] = disp1["Meets 90% Rule"].map(_status)
        st.dataframe(disp1, use_container_width=True, hide_index=True)
        if (~qc["Meets 90% Rule"].astype("boolean").fillna(False)).any():
            st.error("TRUST: One or more periods do not meet 90% payout.")

        st.subheader("Trust Check 2 — (CFO + CFI + CFF + PAT) vs Computed NDCF")
        disp2 = qc[[
            "Financial Year","Period Ended",
            "Cash Flow From operating Activities as per Audited/Reviewed",
            "Cash Flow From Investing Activities as per Audited/Reviewed",
            "Cash Flow From Fincing Activities as per Audited/Reviewed",
            "Profit after tax as per Audited/Reviewed",
        ]].copy()
        # rename for compact headers
        disp2.columns = [
            "Financial Year","Period Ended","CFO","CFI","CFF","PAT"
        ]
        disp2 = disp2.join(qc[["CF Sum","Total Amount of NDCF computed as per NDCF Statement","Gap vs Computed","Gap % of Computed","Within 10% Gap"]])
        disp2["Within 10% Gap"] = disp2["Within 10% Gap"].map(_status)
        st.dataframe(disp2, use_container_width=True, hide_index=True)
        if (~qc["Within 10% Gap"].astype("boolean").fillna(False)).any():
            st.error("TRUST: One or more periods have a gap > 10%.")

        # timeline checks 3a / 3b
        tline = compute_trust_timeline_checks(q)
        st.subheader("Trust Check 3a — Declaration → Record Date (≤ 2 days)")
        t1 = tline[[
            "Financial Year","Period Ended","Declaration Date","Record Date","Days Decl→Record","Record ≤ 2 days"
        ]].copy()
        t1["Record ≤ 2 days"] = t1["Record ≤ 2 days"].map(_status)
        st.dataframe(t1, use_container_width=True, hide_index=True)
        if (tline["Record ≤ 2 days"] == False).any():
            st.error("TRUST: Record Date more than 2 days after Declaration.")

        st.subheader("Trust Check 3b — Record Date → Distribution Date (≤ 5 days)")
        t2 = tline[[
            "Financial Year","Period Ended","Record Date","Distribution Date","Days Record→Distr","Distribution ≤ 5 days"
        ]].copy()
        t2["Distribution ≤ 5 days"] = t2["Distribution ≤ 5 days"].map(_status)
        st.dataframe(t2, use_container_width=True, hide_index=True)
        if (tline["Distribution ≤ 5 days"] == False).any():
            st.error("TRUST: Distribution Date more than 5 days after Record Date.")

        # show RAW strings so we can see what came from the sheet
        with st.expander("Show RAW date strings from sheet"):
            cols = []
            if "Declaration Date__raw" in q.columns: cols.append("Declaration Date__raw")
            if "Record Date__raw" in q.columns: cols.append("Record Date__raw")
            if "Distribution Date__raw" in q.columns: cols.append("Distribution Date__raw")
            st.dataframe(q[cols], use_container_width=True, hide_index=True)

    else:
        if df_spv_all.empty:
            st.info("SPV sheet could not be loaded; skipping SPV checks.")
            return
        q = df_spv_all[(df_spv_all["Name of REIT"] == ent) & (df_spv_all["Financial Year"] == fy)].copy()
        if q.empty:
            st.warning("No SPV rows for the selected REIT and FY.")
            return

        qs = compute_spv_checks(q)

        st.subheader("SPV Check 1 — Declared (incl. Surplus) ≥ 90% of Computed (by SPV/period)")
        d1 = qs[[
            "Name of SPV","Name of Holdco (Leave Blank if N/A)","Financial Year","Period Ended",
            "Total Amount of NDCF computed as per NDCF Statement",
            "Total Amount of NDCF declared for the period (incl. Surplus)",
            "Payout Ratio %","Meets 90% Rule (SPV)"
        ]].copy()
        d1["Meets 90% Rule (SPV)"] = d1["Meets 90% Rule (SPV)"].map(_status)
        st.dataframe(d1, use_container_width=True, hide_index=True)
        if (~qs["Meets 90% Rule (SPV)"].astype("boolean").fillna(False)).any():
            st.error("SPV: One or more SPV periods do not meet 90% payout.")

        st.subheader("SPV Check 2 — |(SPV+HoldCo CFO+CFI+CFF+PAT) − Computed| < Computed")
        d2 = qs[[
            "Name of SPV","Name of Holdco (Leave Blank if N/A)","Financial Year","Period Ended",
            "SPV+HoldCo CF Sum","Total Amount of NDCF computed as per NDCF Statement",
            "Gap vs Computed (SPV)","Gap % of Computed (SPV)","Within Computed Bound (SPV)"
        ]].copy()
        d2["Within Computed Bound (SPV)"] = d2["Within Computed Bound (SPV)"].map(_status)
        st.dataframe(d2, use_container_width=True, hide_index=True)
        if (~qs["Within Computed Bound (SPV)"].astype("boolean").fillna(False)).any():
            st.error("SPV: One or more SPV periods have |Gap| ≥ Computed NDCF.")

def render_ndcf():
    render()

if __name__ == "__main__":
    render()
