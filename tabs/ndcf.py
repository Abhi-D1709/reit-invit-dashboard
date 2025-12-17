# tabs/ndcf.py
from __future__ import annotations

import io
import re
import math
import datetime as dt
from typing import Dict, Tuple
from pandas._libs.tslibs.nattype import NaTType
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Constants (sheet names). Adjust only if your Google Sheet tabs change.
# --------------------------------------------------------------------------------------
TRUST_SHEET_NAME = "NDCF REITs"          # REIT trust-level sheet
SPV_SHEET_NAME   = "NDCF SPV REIT"       # REIT SPV-level sheet
BASIC_DIR_SHEET  = "Sheet5"              # in common.DEFAULT_REIT_DIR_URL (Name of REIT, OD Link)

# Columns we expect (with several tolerant aliases)
COL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Entity": (
        "Entity", "Name of REIT", "REIT", "Trust"
    ),
    "FY": (
        "Financial Year", "Financial Year", "FY"
    ),
    "Period": (
        "Period Ended", "Quarter", "Period"
    ),
    "Decl": (
        "Date of Finalisation/Declaration of NDCF Statement by REIT",
        "Date of Filisation/Declaration of NDCF Statement by REIT",
        "Date of Finalisation of NDCF Statement by REIT",
        "Declaration Date",
    ),
    "Record": (
        "Record Date", "Record date",
    ),
    "Distr": (
        "Date of Distribution of NDCF by REIT",
        "Distribution Date",
    ),
    "Computed": (
        "Total Amount of NDCF computed as per NDCF Statement",
        "Total NDCF computed"
    ),
    "DeclaredIncl": (
        "Total Amount of NDCF declared for the period (incl. Surplus)",
        "Total Amount of NDCF declared (incl. Surplus)",
    ),
    "CFO": (
        "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financals with Limited Review)",
        "CFO",
    ),
    "CFI": (
        "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)",
        "CFI",
    ),
    "CFF": (
        "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials with Limited Review)",
        "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited "
        "s or Financials with Limited Review)",
        "CFF",
    ),
    "PAT": (
        "Profit after tax as per Statement of Profit and Loss (as per Audited Financials with Limited Review)",
        "PAT",
    ),
}

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _gsheet_csv_url(sheet_url: str, *, sheet_name: str) -> str:
    """
    Build a CSV-export URL using the 'gviz' endpoint so we can specify the sheet name.
    """
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)/", sheet_url)
    if not m:
        return sheet_url  # fallback; user might already paste a CSV endpoint
    file_id = m.group(1)
    # gviz allows selecting by sheet name reliably
    return f"https://docs.google.com/spreadsheets/d/{file_id}/gviz/tq?tqx=out:csv&sheet={pd.io.common.quote(sheet_name)}"

def _read_sheet_as_str(sheet_url: str, sheet_name: str) -> pd.DataFrame:
    """
    Always read as string first (prevents Pandas from guessing types).
    """
    url = _gsheet_csv_url(sheet_url, sheet_name=sheet_name)
    return pd.read_csv(url, dtype=str).replace({np.nan: None})

def _normalize_header(h: str | None) -> str:
    if h is None:
        return ""
    # collapse whitespace, lower-case, strip punctuation-likes
    s = re.sub(r"\s+", " ", h).strip().lower()
    s = s.replace("\xa0", " ")  # NBSP -> space
    return s

def _resolve_columns(df: pd.DataFrame, aliases: Dict[str, Tuple[str, ...]]) -> Dict[str, str]:
    """
    Find the best matching real column name in df for each logical key in COL_ALIASES.
    """
    norm_cols = {c: _normalize_header(c) for c in df.columns}
    resolved: Dict[str, str] = {}
    for key, alts in aliases.items():
        match_name = None
        for alt in alts:
            target = _normalize_header(alt)
            for real, norm in norm_cols.items():
                if norm == target:
                    match_name = real
                    break
            if match_name:
                break
        if match_name:
            resolved[key] = match_name
    return resolved

def _coerce_number(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "na", "n/a", "none", "null"):
        return None
    try:
        return float(s)
    except Exception:
        return None

def parse_date_any(v) -> pd.Timestamp | NaTType:
    """
    Robust date parser:
      - trims whitespace / NBSP
      - accepts dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd
      - accepts Excel serials (integers / floats)
      - returns NaT if cannot parse
    """
    if v is None:
        return pd.NaT
    if isinstance(v, (pd.Timestamp, dt.date, dt.datetime)):
        return pd.to_datetime(v)

    s = str(v).replace("\xa0", " ").strip()
    if s == "" or s.lower() in {"na", "n/a", "none", "-", "null"}:
        return pd.NaT

    # Excel serial?
    if re.fullmatch(r"\d{3,6}", s):  # e.g., 45122
        try:
            # Excel uses 1899-12-30 as origin in Google Sheets/Excel CSV exports
            return pd.to_datetime(float(s), unit="D", origin="1899-12-30", utc=False)
        except Exception:
            pass

    # Common day-first patterns
    try:
        # Try pandas first with dayfirst
        return pd.to_datetime(s, dayfirst=True, errors="raise")
    except Exception:
        pass

    # Manual dd/mm/yyyy pattern
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s)
    if m:
        d, mth, y = m.groups()
        y = int(y)
        if y < 100:
            y += 2000
        try:
            return pd.Timestamp(dt.date(y, int(mth), int(d)))
        except Exception:
            return pd.NaT

    # Final attempt (ISO-ish or with month names)
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT

def _pct(n: float | None, d: float | None) -> float | None:
    if n is None or d is None or d == 0:
        return None
    return 100.0 * n / d

def _good_bad_icon(ok: bool | None) -> str:
    if ok is None:
        return "—"
    return "🟢" if ok else "🔴"

# --------------------------------------------------------------------------------------
# Trust-level logic
# --------------------------------------------------------------------------------------

def _prepare_trust(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cols = _resolve_columns(df_raw, COL_ALIASES)

    # keep only columns we know about; tolerate missing ones
    keep = [cols.get(k) for k in ("Entity", "FY", "Period", "Decl", "Record", "Distr",
                                  "Computed", "DeclaredIncl", "CFO", "CFI", "CFF", "PAT")]
    keep = [c for c in keep if c]
    df = df_raw[keep].copy()

    # Standardized column names
    rename_map = {
        cols.get("Entity", ""): "Entity",
        cols.get("FY", ""): "Financial Year",
        cols.get("Period", ""): "Period Ended",
        cols.get("Decl", ""): "Declaration Date (raw)",
        cols.get("Record", ""): "Record Date (raw)",
        cols.get("Distr", ""): "Distribution Date (raw)",
        cols.get("Computed", ""): "Computed NDCF",
        cols.get("DeclaredIncl", ""): "Declared NDCF (incl. Surplus)",
        cols.get("CFO", ""): "CFO",
        cols.get("CFI", ""): "CFI",
        cols.get("CFF", ""): "CFF",
        cols.get("PAT", ""): "PAT",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k}, inplace=True)

    # numeric columns
    for c in ("Computed NDCF", "Declared NDCF (incl. Surplus)", "CFO", "CFI", "CFF", "PAT"):
        if c in df.columns:
            df[c] = df[c].map(_coerce_number)

    # parse dates from raw text columns (keep a copy of raw for diagnostics)
    for col_raw, col_parsed in (
        ("Declaration Date (raw)", "Declaration Date"),
        ("Record Date (raw)", "Record Date"),
        ("Distribution Date (raw)", "Distribution Date"),
    ):
        if col_raw in df.columns:
            df[col_parsed] = df[col_raw].map(parse_date_any)
        else:
            df[col_parsed] = pd.NaT

    # derived calculations
    if {"Computed NDCF", "Declared NDCF (incl. Surplus)"}.issubset(df.columns):
        df["Payout >= 90%"] = df.apply(
            lambda r: (r["Declared NDCF (incl. Surplus)"] is not None and
                       r["Computed NDCF"] not in (None, 0) and
                       _pct(r["Declared NDCF (incl. Surplus)"], r["Computed NDCF"]) >= 90.0),
            axis=1,
        )
        df["Payout Ratio %"] = df.apply(
            lambda r: _pct(r["Declared NDCF (incl. Surplus)"], r["Computed NDCF"]),
            axis=1,
        )

    if {"CFO", "CFI", "CFF", "PAT", "Computed NDCF"}.issubset(df.columns):
        df["CFO+CFI+CFF+PAT"] = df[["CFO", "CFI", "CFF", "PAT"]].sum(axis=1, min_count=1)
        df["Gap % of Computed"] = df.apply(
            lambda r: None if r["Computed NDCF"] in (None, 0)
            else abs(r["CFO+CFI+CFF+PAT"] - r["Computed NDCF"]) / abs(r["Computed NDCF"]) * 100.0,
            axis=1,
        )
        df["Within 10% Gap"] = df["Gap % of Computed"].map(lambda x: (x is not None) and (x <= 10.0))

    # timeline diffs
    df["Days Decl→Record"] = (df["Record Date"] - df["Declaration Date"]).dt.days
    df["Record ≤ 2 days"]   = df["Days Decl→Record"].map(lambda x: (x is not None) and (not pd.isna(x)) and (x <= 2))

    df["Days Record→Distr"] = (df["Distribution Date"] - df["Record Date"]).dt.days
    df["Distribution ≤ 5 days"] = df["Days Record→Distr"].map(lambda x: (x is not None) and (not pd.isna(x)) and (x <= 5))

    return df, cols

# --------------------------------------------------------------------------------------
# SPV-level logic (same date parsing foundation available if needed later)
# --------------------------------------------------------------------------------------

def _prepare_spv(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    # For now we focus on trust-level asks. This keeps the same parsing routine available.
    return df_raw.copy(), {}

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

def _offer_doc_link(entity: str) -> str | None:
    # Pull from BASIC DETAILS directory in common.DEFAULT_REIT_DIR_URL (Sheet5)
    try:
        from utils.common import DEFAULT_REIT_DIR_URL
    except Exception:
        return None
    try:
        dir_df = _read_sheet_as_str(DEFAULT_REIT_DIR_URL, BASIC_DIR_SHEET)
        # normalize headers
        cols = {c.lower().strip(): c for c in dir_df.columns}
        name_col = cols.get("name of reit") or cols.get("entity") or list(dir_df.columns)[0]
        link_col = cols.get("od link") or cols.get("offer document link") or list(dir_df.columns)[-1]
        m = dir_df[dir_df[name_col].str.strip().str.casefold() == entity.strip().casefold()]
        if not m.empty:
            val = (m.iloc[0][link_col] or "").strip()
            return val or None
    except Exception:
        return None
    return None

def render():
    st.header("NDCF")

    # Inputs
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], index=0)
    data_url = st.text_input(
        "Data URL (Google Sheet - public view)",
        value="https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing",
        help="Paste the workbook URL. We will read the proper sheet by name.",
    )

    # Load trust-level REIT data as strings
    trust_raw = _read_sheet_as_str(data_url, TRUST_SHEET_NAME)
    trust_df, cols_map = _prepare_trust(trust_raw)

    # Entity & FY selections (entity first, then FY)
    entities = sorted([e for e in trust_df["Entity"].dropna().unique()])
    entity = st.selectbox("Choose REIT", entities) if entities else None

    if entity:
        od = _offer_doc_link(entity)
        if od:
            st.caption(f"Offer Document: {od}")

        trust_entity = trust_df[trust_df["Entity"].str.strip().str.casefold() == entity.strip().casefold()].copy()
        years = sorted([fy for fy in trust_entity["Financial Year"].dropna().unique()])
        fy = st.selectbox("Financial Year", years) if years else None
    else:
        trust_entity, fy = pd.DataFrame(), None

    if entity and fy:
        q = trust_entity[trust_entity["Financial Year"] == fy].copy()

        # --------- Trust Check 1 (90% payout) ----------
        st.subheader("Trust Check 1 — 90% payout of Computed NDCF")
        if q.empty:
            st.info("No rows for the selected REIT/FY.")
        else:
            view = q[[
                "Financial Year", "Period Ended", "Computed NDCF", "Declared NDCF (incl. Surplus)",
                "Payout Ratio %", "Payout >= 90%"
            ]].copy()
            view["Payout Ratio %"] = view["Payout Ratio %"].map(lambda x: None if x is None else round(x, 2))
            view["Payout >= 90%"] = view["Payout >= 90%"].map(_good_bad_icon)
            st.dataframe(view, use_container_width=True)
            if (~q["Payout >= 90%"].fillna(False)).any():
                st.error("TRUST: One or more periods are below 90% payout.")
            else:
                st.success("TRUST: All periods meet ≥ 90% payout.")

        # --------- Trust Check 2 (CFO+CFI+CFF+PAT vs Computed within 10%) ----------
        st.subheader("Trust Check 2 — (CFO + CFI + CFF + PAT) vs Computed NDCF")
        if not q.empty:
            view2 = q[[
                "Financial Year", "Period Ended", "CFO", "CFI", "CFF", "PAT",
                "CFO+CFI+CFF+PAT", "Computed NDCF", "Gap % of Computed", "Within 10% Gap"
            ]].copy()
            for c in ("Gap % of Computed",):
                view2[c] = view2[c].map(lambda x: None if x is None else round(x, 2))
            view2["Within 10% Gap"] = view2["Within 10% Gap"].map(_good_bad_icon)
            st.dataframe(view2, use_container_width=True)
            if (~q["Within 10% Gap"].fillna(False)).any():
                st.error("TRUST: One or more periods have a gap > 10% between (CFO + CFI + CFF + PAT) and Computed NDCF.")
            else:
                st.success("TRUST: All periods are within the 10% gap threshold.")

        # --------- Trust Check 3a (Declaration → Record ≤ 2 days) ----------
        st.subheader("Trust Check 3a — Declaration → Record Date (≤ 2 days)")
        if not q.empty:
            view3a = q[[
                "Financial Year", "Period Ended",
                "Declaration Date", "Record Date",
                "Days Decl→Record", "Record ≤ 2 days"
            ]].copy()
            view3a["Record ≤ 2 days"] = view3a["Record ≤ 2 days"].map(_good_bad_icon)
            st.dataframe(view3a, use_container_width=True)

            # raw diagnostics (show strings as they appear in the sheet)
            with st.expander("Show raw date values (diagnostics)"):
                raw = q[[c for c in q.columns if c.endswith("(raw)")]].copy()
                raw.rename(columns={
                    "Declaration Date (raw)": "Declaration (raw)",
                    "Record Date (raw)": "Record (raw)",
                    "Distribution Date (raw)": "Distribution (raw)",
                }, inplace=True)
                st.dataframe(raw, use_container_width=True)

            if (~q["Record ≤ 2 days"].fillna(False)).any():
                st.error("TRUST: One or more periods have Record Date more than 2 days after Declaration.")
            else:
                st.success("TRUST: All periods meet ≤ 2 days from Declaration to Record Date.")

        # --------- Trust Check 3b (Record → Distribution ≤ 5 days) ----------
        st.subheader("Trust Check 3b — Record Date → Distribution Date (≤ 5 days)")
        if not q.empty:
            view3b = q[[
                "Financial Year", "Period Ended",
                "Record Date", "Distribution Date",
                "Days Record→Distr", "Distribution ≤ 5 days"
            ]].copy()
            view3b["Distribution ≤ 5 days"] = view3b["Distribution ≤ 5 days"].map(_good_bad_icon)
            st.dataframe(view3b, use_container_width=True)

            if (~q["Distribution ≤ 5 days"].fillna(False)).any():
                st.error("TRUST: One or more periods have Distribution Date more than 5 days after Record Date.")
            else:
                st.success("TRUST: All periods meet ≤ 5 days from Record Date to Distribution Date.")
    else:
        st.info("Select a REIT and a Financial Year to view checks.")
