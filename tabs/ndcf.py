# tabs/ndcf.py
import re
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------- Defaults / wiring ------------------------------
DEFAULT_SHEET_URL_TRUST = (
    "https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing"
)
TRUST_SHEET_NAME = "NDCF REITs"
SPV_SHEET_NAME = "NDCF SPV REIT"

# Offer-document workbook (Sheet5) for OD links
DEFAULT_REIT_DIR_URL: Optional[str] = None

# If your utils.common defines central constants, pick them up
try:
    from utils.common import (
        NDCF_REITS_SHEET_URL as _URL_TRUST,
        DEFAULT_REIT_DIR_URL as _DIR_URL,
    )

    if _URL_TRUST:
        DEFAULT_SHEET_URL_TRUST = _URL_TRUST
    if _DIR_URL:
        DEFAULT_REIT_DIR_URL = _DIR_URL
except Exception:
    pass


# ------------------------------- Tiny utilities ------------------------------
def _strip(s):
    return str(s).strip() if pd.notna(s) else s


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


def _excel_serial_to_date(n: float) -> pd.Timestamp:
    """Excel 1900-based serial; 1899-12-30 base handles the leap bug."""
    try:
        n_float = float(n)
    except Exception:
        return pd.NaT
    if 2 <= n_float < 100000:
        base = pd.Timestamp("1899-12-30")
        try:
            return base + pd.to_timedelta(int(round(n_float)), unit="D")
        except Exception:
            return pd.NaT
    return pd.NaT


def _to_date(v) -> pd.Timestamp:
    """
    Parse dates from:
      - plain strings (dd/mm/yyyy, dd-mm-yyyy, etc.; dayfirst=True)
      - JS gviz 'Date(YYYY,MM,DD,...)' strings (month 0-based)
      - Excel serial numbers
    """
    if pd.isna(v):
        return pd.NaT

    if isinstance(v, (int, float, np.integer, np.floating)):
        dt = _excel_serial_to_date(v)
        if pd.notna(dt):
            return dt

    s = str(v).strip()

    m = re.match(r"^Date\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", s)
    if m:
        y, mth, d = map(int, m.groups())
        try:
            return pd.Timestamp(year=y, month=mth + 1, day=d)
        except Exception:
            return pd.NaT

    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return pd.NaT
        return pd.to_datetime(dt.date())
    except Exception:
        return pd.NaT


def _status(v: Optional[bool]) -> str:
    if pd.isna(v):
        return "—"
    return "🟢" if bool(v) else "🔴"


def _csv_url_from_gsheet(url: str, *, sheet: Optional[str] = None, gid: Optional[str] = None) -> str:
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


# ------------------------------- Loaders -------------------------------------
def _read_trust_df_from_gsheet(sheet_url: str) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(sheet_url or DEFAULT_SHEET_URL_TRUST, sheet=TRUST_SHEET_NAME)
    df = pd.read_csv(csv_url, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Canonicalize column names (handle spelling variants)
    rename_map = {
        "Entity": "Name of REIT",
        "Financial Year": "Financial Year",
        "Fincial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Date of Filisation/Declaration of NDCF Statement by REIT": "Declaration Date",
        "Date of Finalisation/Declaration of NDCF Statement by REIT": "Declaration Date",
        "Date of Finalization/Declaration of NDCF Statement by REIT": "Declaration Date",
        "Record Date": "Record Date",
        "Date of Distribution of NDCF by REIT": "Distribution Date",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Fuzzy fallback
    def _pick(cols, *tokens) -> Optional[str]:
        for c in cols:
            cl = c.lower()
            for a, b in [
                ("filisation", "finalisation"),
                ("finalisation", "finalization"),
                ("finacial", "financial"),
                ("fincial", "financial"),
            ]:
                cl = cl.replace(a, b)
            if all(t in cl for t in tokens):
                return c
        return None

    if "Declaration Date" not in df.columns:
        cand = _pick(df.columns, "declar") or _pick(df.columns, "finaliz", "ndcf")
        if cand:
            df.rename(columns={cand: "Declaration Date"}, inplace=True)
    if "Record Date" not in df.columns:
        cand = _pick(df.columns, "record", "date")
        if cand:
            df.rename(columns={cand: "Record Date"}, inplace=True)
    if "Distribution Date" not in df.columns:
        cand = _pick(df.columns, "distribution", "date")
        if cand:
            df.rename(columns={cand: "Distribution Date"}, inplace=True)

    # Required numeric and id columns
    needed = [
        "Name of REIT",
        "Financial Year",
        "Period Ended",
        "Total Amount of NDCF computed as per NDCF Statement",
        "Total Amount of NDCF declared for the period (incl. Surplus)",
        "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error("The NDCF (Trust) sheet is missing columns: " + ", ".join(missing))
        with st.expander("Show detected columns (Trust)"):
            st.write(list(df.columns))
        return df.iloc[0:0]

    for c in needed[3:]:
        df[c] = df[c].map(_to_number)

    for c in ["Name of REIT", "Financial Year", "Period Ended"]:
        df[c] = df[c].astype(str).map(_strip)

    for c in ["Declaration Date", "Record Date", "Distribution Date"]:
        if c in df.columns:
            df[c] = df[c].map(_to_date)

    return df


def _read_spv_df_from_gsheet(sheet_url: str) -> pd.DataFrame:
    csv_url = _csv_url_from_gsheet(sheet_url or DEFAULT_SHEET_URL_TRUST, sheet=SPV_SHEET_NAME)
    df = pd.read_csv(csv_url, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Entity": "Name of REIT",
        "Financial Year": "Financial Year",
        "Fincial Year": "Financial Year",
        "Period": "Period Ended",
        "Period ended": "Period Ended",
        "Name of Holdco": "Name of Holdco (Leave Blank if N/A)",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    needed = [
        "Name of REIT",
        "Name of SPV",
        "Name of Holdco (Leave Blank if N/A)",
        "Financial Year",
        "Period Ended",
        "Total Amount of NDCF computed as per NDCF Statement",
        "Total Amount of NDCF declared for the period (incl. Surplus)",
        "SPV Cash Flow From operating Activities as per Audited/Reviewed",
        "SPV Cash Flow From Investing Activities as per Audited/Reviewed",
        "SPV Cash Flow From Financing Activities as per Audited/Reviewed",
        "SPV Profit after tax as per Audited/Reviewed",
        "HoldCo Cash Flow From operating Activities as per Audited/Reviewed",
        "HoldCo Cash Flow From Investing Activities as per Audited/Reviewed",
        "Holdco Cash Flow From Financing Activities as per Audited/Reviewed",
        "Holdco Profit after tax as per Audited/Reviewed",
    ]

    # Backward-compatible fallbacks for longer headers
    long_to_short = {
        "SPV Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)": "SPV Cash Flow From operating Activities as per Audited/Reviewed",
        "SPV Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)": "SPV Cash Flow From Investing Activities as per Audited/Reviewed",
        "SPV Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)": "SPV Cash Flow From Financing Activities as per Audited/Reviewed",
        "SPV Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)": "SPV Profit after tax as per Audited/Reviewed",
        "HoldCo Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)": "HoldCo Cash Flow From operating Activities as per Audited/Reviewed",
        "HoldCo Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)": "HoldCo Cash Flow From Investing Activities as per Audited/Reviewed",
        "Holdco Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)": "Holdco Cash Flow From Financing Activities as per Audited/Reviewed",
        "Holdco Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)": "Holdco Profit after tax as per Audited/Reviewed",
    }
    for k, v in long_to_short.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning("The NDCF (SPV) sheet is missing columns: " + ", ".join(missing))
        with st.expander("Show detected columns (SPV)"):
            st.write(list(df.columns))
        return df.iloc[0:0]

    for c in needed[5:]:
        df[c] = df[c].map(_to_number)

    for c in ["Name of REIT", "Financial Year", "Period Ended", "Name of SPV", "Name of Holdco (Leave Blank if N/A)"]:
        df[c] = df[c].astype(str).map(_strip)

    return df


def _load_offer_doc_links(dir_url: Optional[str]) -> pd.DataFrame:
    if not dir_url:
        return pd.DataFrame(columns=["Name of REIT", "OD Link"])
    try:
        csv_url = _csv_url_from_gsheet(dir_url, sheet="Sheet5")
        df = pd.read_csv(csv_url, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        ent_col = next((c for c in df.columns if "name" in c.lower() and "reit" in c.lower()), None)
        link_col = next((c for c in df.columns if "od" in c.lower() and "link" in c.lower()), None)
        if not ent_col:
            ent_col = "Name of REIT" if "Name of REIT" in df.columns else df.columns[0]
        if not link_col:
            for c in df.columns:
                if "link" in c.lower():
                    link_col = c
                    break
            if not link_col:
                link_col = df.columns[-1]
        return df[[ent_col, link_col]].rename(columns={ent_col: "Name of REIT", link_col: "OD Link"})
    except Exception:
        return pd.DataFrame(columns=["Name of REIT", "OD Link"])


# ------------------------------- Calculations --------------------------------
def compute_trust_checks(df: pd.DataFrame) -> pd.DataFrame:
    comp = "Total Amount of NDCF computed as per NDCF Statement"
    decl = "Total Amount of NDCF declared for the period (incl. Surplus)"
    cfo = "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    cfi = "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    cff = "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    pat = "Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"

    out = df.copy()
    out["Payout Ratio %"] = np.where(out[comp] > 0, (out[decl] / out[comp]) * 100.0, np.nan).round(2)
    out["Meets 90% Rule"] = out["Payout Ratio %"] >= 90.0

    out["CF Sum"] = out[cfo].fillna(0) + out[cfi].fillna(0) + out[cff].fillna(0) + out[pat].fillna(0)
    out["Gap vs Computed"] = out["CF Sum"] - out[comp]
    out["Gap % of Computed"] = np.where(out[comp] != 0, (out["Gap vs Computed"] / out[comp]) * 100.0, np.nan).round(2)
    out["Within 10% Gap"] = out["Gap % of Computed"].abs() <= 10.0
    return out


def compute_trust_timeline_checks(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Declaration Date", "Record Date", "Distribution Date"}.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "Financial Year",
                "Period Ended",
                "Declaration Date",
                "Record Date",
                "Distribution Date",
                "Days Decl→Record",
                "Record ≤ 2 days",
                "Days Record→Distr",
                "Distribution ≤ 5 days",
            ]
        )
    t = df.copy()
    t["Days Decl→Record"] = (t["Record Date"] - t["Declaration Date"]).dt.days
    t["Days Record→Distr"] = (t["Distribution Date"] - t["Record Date"]).dt.days
    t["Record ≤ 2 days"] = (t["Days Decl→Record"] >= 0) & (t["Days Decl→Record"] <= 2)
    t["Distribution ≤ 5 days"] = (t["Days Record→Distr"] >= 0) & (t["Days Record→Distr"] <= 5)
    return t[
        [
            "Financial Year",
            "Period Ended",
            "Declaration Date",
            "Record Date",
            "Distribution Date",
            "Days Decl→Record",
            "Record ≤ 2 days",
            "Days Record→Distr",
            "Distribution ≤ 5 days",
        ]
    ].copy()


def compute_spv_checks(df: pd.DataFrame) -> pd.DataFrame:
    comp = "Total Amount of NDCF computed as per NDCF Statement"
    decl = "Total Amount of NDCF declared for the period (incl. Surplus)"

    spv_cfo = "SPV Cash Flow From operating Activities as per Audited/Reviewed"
    spv_cfi = "SPV Cash Flow From Investing Activities as per Audited/Reviewed"
    spv_cff = "SPV Cash Flow From Financing Activities as per Audited/Reviewed"
    spv_pat = "SPV Profit after tax as per Audited/Reviewed"

    hco_cfo = "HoldCo Cash Flow From operating Activities as per Audited/Reviewed"
    hco_cfi = "HoldCo Cash Flow From Investing Activities as per Audited/Reviewed"
    hco_cff = "Holdco Cash Flow From Financing Activities as per Audited/Reviewed"
    hco_pat = "Holdco Profit after tax as per Audited/Reviewed"

    out = df.copy()
    out["Payout Ratio %"] = np.where(out[comp] > 0, (out[decl] / out[comp]) * 100.0, np.nan).round(2)
    out["Meets 90% Rule (SPV)"] = out["Payout Ratio %"] >= 90.0

    out["SPV+HoldCo CF Sum"] = (
        out[spv_cfo].fillna(0)
        + out[spv_cfi].fillna(0)
        + out[spv_cff].fillna(0)
        + out[spv_pat].fillna(0)
        + out[hco_cfo].fillna(0)
        + out[hco_cfi].fillna(0)
        + out[hco_cff].fillna(0)
        + out[hco_pat].fillna(0)
    )
    out["Gap vs Computed (SPV)"] = out["SPV+HoldCo CF Sum"] - out[comp]
    out["Gap % of Computed (SPV)"] = np.where(
        out[comp] != 0, (out["Gap vs Computed (SPV)"] / out[comp]) * 100.0, np.nan
    ).round(2)
    out["Within Computed Bound (SPV)"] = np.where(out[comp] > 0, out["Gap vs Computed (SPV)"].abs() < out[comp], np.nan)
    return out


# --------------------------------- UI ----------------------------------------
def render():
    st.header("NDCF — Compliance Checks")

    with st.sidebar:
        seg = st.selectbox("Select Segment", ["REIT", "InvIT"], index=0)
    
    # Auto-set URL (Hidden)
    gsheet_url = DEFAULT_SHEET_URL_TRUST

    if seg != "REIT":
        st.info("InvIT checks will be added later.")
        return

    # Load from Google Sheet only (as requested)
    df_trust_all = _read_trust_df_from_gsheet(gsheet_url)
    df_spv_all = _read_spv_df_from_gsheet(gsheet_url)

    if df_trust_all.empty:
        return

    ent = st.sidebar.selectbox(
        "Choose REIT",
        sorted(df_trust_all["Name of REIT"].dropna().unique().tolist()),
        index=0,
        key="ndcf_reit_select",
    )

    # Offer document link (Sheet5 in Default REIT Directory workbook)
    if DEFAULT_REIT_DIR_URL:
        od_df = _load_offer_doc_links(DEFAULT_REIT_DIR_URL)
        link = od_df.loc[od_df["Name of REIT"] == ent, "OD Link"]
        if not link.empty and isinstance(link.iloc[0], str) and link.iloc[0].strip():
            st.markdown(f"**Offer Document:** [{link.iloc[0].strip()}]({link.iloc[0].strip()})")

    level = st.sidebar.radio("Analysis level", ["Trust", "SPV/HoldCo"], horizontal=True, key="ndcf_level_select")

    if level == "Trust":
        fy_options = sorted(
            df_trust_all.loc[df_trust_all["Name of REIT"] == ent, "Financial Year"].dropna().unique().tolist()
        )
    else:
        fy_options = [] if df_spv_all.empty else sorted(
            df_spv_all.loc[df_spv_all["Name of REIT"] == ent, "Financial Year"].dropna().unique().tolist()
        )

    fy = st.sidebar.selectbox("Financial Year", ["— Select —"] + fy_options, index=0, key="ndcf_fy_select")

    if fy == "— Select —":
        st.info("Pick a Financial Year to show results.")
        return

    # --------------------------- TRUST LEVEL ----------------------------------
    if level == "Trust":
        q = df_trust_all[(df_trust_all["Name of REIT"] == ent) & (df_trust_all["Financial Year"] == fy)].copy()
        if q.empty:
            st.warning("No TRUST-level rows for the selected REIT and Financial Year.")
            return

        q = compute_trust_checks(q)

        total = int(len(q))
        good_payout = int(q["Meets 90% Rule"].astype("boolean").fillna(False).sum())
        good_gap = int(q["Within 10% Gap"].astype("boolean").fillna(False).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("TRUST: periods meeting 90% payout", f"{good_payout}/{total}")
        c2.metric("TRUST: periods within 10% gap", f"{good_gap}/{total}")
        c3.metric("TRUST: rows analysed", f"{total}")

        st.subheader("Trust Check 1 — 90% payout of Computed NDCF (period-wise)")
        disp1 = q[
            [
                "Financial Year",
                "Period Ended",
                "Total Amount of NDCF computed as per NDCF Statement",
                "Total Amount of NDCF declared for the period (incl. Surplus)",
                "Payout Ratio %",
                "Meets 90% Rule",
            ]
        ].copy()
        disp1["Meets 90% Rule"] = disp1["Meets 90% Rule"].map(_status)
        st.dataframe(disp1, use_container_width=True, hide_index=True)
        if (~q["Meets 90% Rule"].astype("boolean").fillna(False)).any():
            st.error(
                "TRUST: One or more periods do **not** meet the 90% payout requirement (Declared incl. surplus < 90% of Computed NDCF)."
            )

        st.subheader("Trust Check 2 — (CFO + CFI + CFF + PAT) gap vs Computed NDCF (period-wise)")
        disp2 = q[
            [
                "Financial Year",
                "Period Ended",
                "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
                "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
                "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
                "Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)",
                "CF Sum",
                "Total Amount of NDCF computed as per NDCF Statement",
                "Gap vs Computed",
                "Gap % of Computed",
                "Within 10% Gap",
            ]
        ].copy()
        disp2["Within 10% Gap"] = disp2["Within 10% Gap"].map(_status)
        st.dataframe(disp2, use_container_width=True, hide_index=True)
        if (~q["Within 10% Gap"].astype("boolean").fillna(False)).any():
            st.error("TRUST: One or more periods have a gap **> 10%** between (CFO + CFI + CFF + PAT) and Computed NDCF.")

        # -------- Split timeline checks into two separate tables ----------
        tline = compute_trust_timeline_checks(q)
        if tline.empty:
            st.info("Declaration / Record / Distribution columns not found; timeline checks skipped.")
        else:
            st.subheader("Trust Check 3 — Declaration → Record Date (≤ 2 days)")
            t1 = tline[
                ["Financial Year", "Period Ended", "Declaration Date", "Record Date", "Days Decl→Record", "Record ≤ 2 days"]
            ].copy()
            t1["Record ≤ 2 days"] = t1["Record ≤ 2 days"].map(_status)
            st.dataframe(t1, use_container_width=True, hide_index=True)
            if (tline["Record ≤ 2 days"] == False).any():
                st.error("TRUST: One or more periods have **Record Date more than 2 days after Declaration**.")

            st.subheader("Trust Check 4 — Record Date → Distribution Date (≤ 5 days)")
            t2 = tline[
                ["Financial Year", "Period Ended", "Record Date", "Distribution Date", "Days Record→Distr", "Distribution ≤ 5 days"]
            ].copy()
            t2["Distribution ≤ 5 days"] = t2["Distribution ≤ 5 days"].map(_status)
            st.dataframe(t2, use_container_width=True, hide_index=True)
            if (tline["Distribution ≤ 5 days"] == False).any():
                st.error("TRUST: One or more periods have **Distribution Date more than 5 days after Record Date**.")

    # ------------------------------ SPV LEVEL ---------------------------------
    else:
        if df_spv_all.empty:
            st.info("SPV sheet could not be loaded or columns are missing; skipping SPV checks.")
            return

        q = df_spv_all[(df_spv_all["Name of REIT"] == ent) & (df_spv_all["Financial Year"] == fy)].copy()
        if q.empty:
            st.warning("No SPV-level rows for the selected REIT and Financial Year.")
            return

        q = compute_spv_checks(q)

        st.subheader("SPV Check 1 — Declared (incl. Surplus) ≥ 90% of Computed NDCF (by SPV/period)")
        disp_s1 = q[
            [
                "Name of SPV",
                "Name of Holdco (Leave Blank if N/A)",
                "Financial Year",
                "Period Ended",
                "Total Amount of NDCF computed as per NDCF Statement",
                "Total Amount of NDCF declared for the period (incl. Surplus)",
                "Payout Ratio %",
                "Meets 90% Rule (SPV)",
            ]
        ].copy()
        disp_s1["Meets 90% Rule (SPV)"] = disp_s1["Meets 90% Rule (SPV)"].map(_status)
        st.dataframe(disp_s1, use_container_width=True, hide_index=True)
        if (~q["Meets 90% Rule (SPV)"].astype("boolean").fillna(False)).any():
            st.error("SPV: One or more SPV periods do **not** meet the 90% payout requirement.")

        st.subheader("SPV Check 2 — |(SPV+HoldCo CFO+CFI+CFF+PAT) − Computed NDCF| < Computed NDCF")
        disp_s2 = q[
            [
                "Name of SPV",
                "Name of Holdco (Leave Blank if N/A)",
                "Financial Year",
                "Period Ended",
                "SPV+HoldCo CF Sum",
                "Total Amount of NDCF computed as per NDCF Statement",
                "Gap vs Computed (SPV)",
                "Gap % of Computed (SPV)",
                "Within Computed Bound (SPV)",
            ]
        ].copy()
        disp_s2["Within Computed Bound (SPV)"] = disp_s2["Within Computed Bound (SPV)"].map(_status)
        st.dataframe(disp_s2, use_container_width=True, hide_index=True)
        if (~q["Within Computed Bound (SPV)"].astype("boolean").fillna(False)).any():
            st.error("SPV: One or more SPV periods have |Gap| ≥ Computed NDCF.")


# Entry point used by pages/5_NDCF.py
def render_ndcf():
    render()


if __name__ == "__main__":
    render()
