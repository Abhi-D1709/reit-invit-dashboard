# tabs/ndcf.py
import re
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils import rules, status  # every threshold lives in utils/rules.py

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


def _tri(passed: pd.Series, unknown: pd.Series) -> pd.Series:
    """Nullable boolean: True/False where the inputs exist, <NA> ("insufficient data") where they
    don't. A comparison against a missing value is False in plain pandas, which reported missing
    data as a failed check."""
    return passed.astype("boolean").mask(unknown)


def _status(v: Optional[bool]) -> str:
    return status.tag(status.from_bool(v))


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
    out["Meets payout rule"] = _tri(out["Payout Ratio %"] >= rules.NDCF_PAYOUT_MIN_PCT, out["Payout Ratio %"].isna())

    # all four figures are needed; a missing one used to count as 0 and gave a made-up total
    out["CF Sum"] = out[[cfo, cfi, cff, pat]].sum(axis=1, min_count=4)
    out["Gap vs Computed"] = out["CF Sum"] - out[comp]
    out["Gap % of Computed"] = np.where(out[comp] != 0, (out["Gap vs Computed"] / out[comp]) * 100.0, np.nan).round(2)
    out["Within gap limit"] = _tri(out["Gap % of Computed"].abs() <= rules.NDCF_CF_GAP_MAX_PCT, out["Gap % of Computed"].isna())
    return out


TIMELINE_COLUMNS = [
    "Financial Year", "Period Ended", "Declaration Date", "Record Date", "Distribution Date", "Rule applied",
    "Days Decl→Record", "Working days Decl→Record", "Record on time",
    "Days Record→Distr", "Working days Record→Distr", "Distribution on time",
    "Days Decl→Distr", "Distribution within limit", "Date check",
]
RULE_OLD = "15 days from declaration"
RULE_NEW = "2 + 5 working days"


def working_days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Working days (Monday to Friday) after `start` up to and including `end`; negative when `end` is earlier.
    NaN when either date is missing. Market holidays are not excluded (no holiday calendar is kept)."""
    valid = start.notna() & end.notna()
    a = start.where(valid, pd.Timestamp("2000-01-03")).values.astype("datetime64[D]")
    b = end.where(valid, pd.Timestamp("2000-01-03")).values.astype("datetime64[D]")
    one = np.timedelta64(1, "D")
    counted = np.busday_count(a + one, b + one)  # (start, end] == [start+1, end+1)
    return pd.Series(counted, index=start.index).astype(float).where(valid)


def _new_timeline_applies(declaration) -> bool:
    """Distributions declared on or after the amendment date follow the working-day timeline."""
    return bool(pd.notna(declaration) and declaration.date() >= rules.NDCF_NEW_TIMELINE_FROM)


def _wd(start, end) -> float:
    return float(working_days_between(pd.Series([start]), pd.Series([end])).iloc[0])


def _timeline_is_sound(declaration, record, distribution) -> bool:
    """Dates in order, and within whichever timeline applies to this declaration date."""
    if not declaration <= record <= distribution:
        return False
    if _new_timeline_applies(declaration):
        return _wd(declaration, record) <= rules.NDCF_RECORD_MAX_WORKING_DAYS and \
            _wd(record, distribution) <= rules.NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS
    return (distribution - declaration).days <= rules.NDCF_DISTRIBUTION_MAX_DAYS


def _swap_day_month(ts):
    """The same date with day and month exchanged, or None when that isn't a valid date."""
    if pd.isna(ts) or ts.day > 12:
        return None
    return pd.Timestamp(year=ts.year, month=ts.day, day=ts.month)


def _swap_would_fix(declaration, record, distribution) -> bool:
    """Would exchanging day and month on one or more of the three dates give a sound timeline?
    (Dates typed as dd/mm in some cells and mm/dd in others are common in the source sheet.)"""
    import itertools

    options = [[d] + ([_swap_day_month(d)] if _swap_day_month(d) is not None else []) for d in (declaration, record, distribution)]
    return any(
        combo != (declaration, record, distribution) and _timeline_is_sound(*combo) for combo in itertools.product(*options)
    )


def _holiday_note(row) -> str:
    """A small overrun of a working-day limit may just be a market holiday in the window."""
    over = max(
        row["Working days Decl→Record"] - rules.NDCF_RECORD_MAX_WORKING_DAYS,
        row["Working days Record→Distr"] - rules.NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS,
    )
    if 0 < over <= 2:
        return f"Over the limit by {over:g} working day(s): a market holiday in the window, or a different way of counting the days, could explain it"
    return ""


def _date_check(row) -> str:
    dates = (row["Declaration Date"], row["Record Date"], row["Distribution Date"])
    if any(pd.isna(d) for d in dates):
        return "Missing date(s)"
    hint = " (a day/month swap would fix it)" if _swap_would_fix(*dates) else ""
    days = (row["Days Decl→Record"], row["Days Record→Distr"], row["Days Decl→Distr"])
    if any(d < 0 for d in days):
        return "Dates out of order: check the sheet" + hint
    if hint and not _timeline_is_sound(*dates):
        return "Late as entered" + hint
    if _new_timeline_applies(dates[0]):
        return _holiday_note(row)
    return ""


def _gap_check(days: pd.Series, limit: float) -> pd.Series:
    """True/False for a real gap. <NA> when a date is missing or the dates are out of order (a negative
    gap): that is a data-entry error, not a late payment, and must not be reported as one."""
    return _tri((days >= 0) & (days <= limit), days.isna() | (days < 0))


def compute_trust_timeline_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Reg. 18(16)(c). Declared before rules.NDCF_NEW_TIMELINE_FROM: paid within 15 days of declaration.
    Declared on or after it: record date within 2 working days of declaration and payment within 5 working
    days of the record date. The checks of the other timeline are <NA> (not applicable)."""
    if not {"Declaration Date", "Record Date", "Distribution Date"}.issubset(df.columns):
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    t = df.copy()
    t["Days Decl→Record"] = (t["Record Date"] - t["Declaration Date"]).dt.days
    t["Days Record→Distr"] = (t["Distribution Date"] - t["Record Date"]).dt.days
    t["Days Decl→Distr"] = (t["Distribution Date"] - t["Declaration Date"]).dt.days
    t["Working days Decl→Record"] = working_days_between(t["Declaration Date"], t["Record Date"])
    t["Working days Record→Distr"] = working_days_between(t["Record Date"], t["Distribution Date"])
    # If any two of the three dates are out of order the whole timeline is unreliable, so none of the
    # three checks gives a verdict (which date is wrong is unknown).
    out_of_order = (t[["Days Decl→Record", "Days Record→Distr", "Days Decl→Distr"]] < 0).any(axis=1)
    new = t["Declaration Date"].map(_new_timeline_applies).astype(bool)
    has_rule = t["Declaration Date"].notna()
    t["Rule applied"] = np.where(~has_rule, "", np.where(new, RULE_NEW, RULE_OLD))
    t["Record on time"] = _gap_check(t["Working days Decl→Record"], rules.NDCF_RECORD_MAX_WORKING_DAYS).mask(out_of_order | ~new)
    t["Distribution on time"] = _gap_check(
        t["Working days Record→Distr"], rules.NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS
    ).mask(out_of_order | ~new)
    t["Distribution within limit"] = _gap_check(t["Days Decl→Distr"], rules.NDCF_DISTRIBUTION_MAX_DAYS).mask(out_of_order | new)
    t["Date check"] = t.apply(_date_check, axis=1) if len(t) else pd.Series(dtype="object")
    return t[TIMELINE_COLUMNS].copy()


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
    out["Meets payout rule (SPV)"] = _tri(out["Payout Ratio %"] >= rules.NDCF_PAYOUT_MIN_PCT, out["Payout Ratio %"].isna())

    # The SPV's four figures are always needed. The HoldCo's four are needed only when there is a
    # HoldCo ("Leave Blank if N/A"); without one they contribute nothing.
    holdco_name = out["Name of Holdco (Leave Blank if N/A)"].astype(str).str.strip().str.lower()
    has_holdco = ~holdco_name.isin(["", "nan", "-", "na", "n/a", "none", "nil"])
    spv_sum = out[[spv_cfo, spv_cfi, spv_cff, spv_pat]].sum(axis=1, min_count=4)
    hco_sum = out[[hco_cfo, hco_cfi, hco_cff, hco_pat]].sum(axis=1, min_count=4).where(has_holdco, 0.0)
    out["SPV+HoldCo CF Sum"] = spv_sum + hco_sum
    out["Gap vs Computed (SPV)"] = out["SPV+HoldCo CF Sum"] - out[comp]
    out["Gap % of Computed (SPV)"] = np.where(
        out[comp] != 0, (out["Gap vs Computed (SPV)"] / out[comp]) * 100.0, np.nan
    ).round(2)
    out["Within Computed Bound (SPV)"] = _tri(
        out["Gap vs Computed (SPV)"].abs() < out[comp], out["Gap vs Computed (SPV)"].isna() | ~(out[comp] > 0)
    )
    return out


AREA = "NDCF distribution"


def _is_false(v) -> bool:
    """A real failure: False, not <NA> (which means the check could not run)."""
    return not pd.isna(v) and not bool(v)


def summary_results(entity: str) -> list:
    """Scorecard verdicts for a REIT's most recently declared distribution: payout, timeline, and the
    cash-flow gap (a house rule, so a miss is a Review, not a Fail)."""
    df = _read_trust_df_from_gsheet(DEFAULT_SHEET_URL_TRUST)
    rows = df[df["Name of REIT"] == entity] if not df.empty else df
    if rows.empty:
        return [status.CheckResult("NDCF distribution", status.Status.NO_DATA, "No NDCF rows for this entity", AREA)]
    q = compute_trust_checks(rows).reset_index(drop=True)
    tl = compute_trust_timeline_checks(q).reset_index(drop=True)
    declared = tl["Declaration Date"]
    pos = int(declared.idxmax()) if declared.notna().any() else len(q) - 1
    row, t = q.iloc[pos], tl.iloc[pos]
    period = f"{row['Financial Year']} {row['Period Ended']}"
    Status, CheckResult = status.Status, status.CheckResult

    ratio = row["Payout Ratio %"]
    payout = CheckResult(
        f"Payout ≥ {rules.NDCF_PAYOUT_MIN_PCT:g}% of NDCF", status.from_bool(row["Meets payout rule"]),
        f"{period}: " + ("payout could not be worked out (figures missing)" if pd.isna(ratio) else f"declared {ratio:.1f}% of computed NDCF"),
        AREA, "ndcf.payout_min")

    checks = t[["Record on time", "Distribution on time", "Distribution within limit"]]
    applicable = [status.from_bool(v) for v in checks if not pd.isna(v)]
    if applicable:
        timeline_status = status.worst(applicable)
        late = []
        if _is_false(t["Record on time"]):
            late.append(f"record date {int(t['Working days Decl→Record'])} working days after declaration (limit {rules.NDCF_RECORD_MAX_WORKING_DAYS})")
        if _is_false(t["Distribution on time"]):
            late.append(f"paid {int(t['Working days Record→Distr'])} working days after the record date (limit {rules.NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS})")
        if _is_false(t["Distribution within limit"]):
            late.append(f"paid {int(t['Days Decl→Distr'])} days after declaration (limit {rules.NDCF_DISTRIBUTION_MAX_DAYS})")
        detail = f"{t['Rule applied']}: " + ("; ".join(late) if late else f"paid {int(t['Days Decl→Distr'])} days after declaration, within the limits")
    else:
        timeline_status, detail = Status.NO_DATA, t["Date check"] or "dates missing"
    timeline = CheckResult("Distribution timeline", timeline_status, f"{period}: {detail}", AREA,
                           "ndcf.record_working_days" if t["Rule applied"] == RULE_NEW else "ndcf.distribution_days")

    gap = row["Within gap limit"]
    gap_result = CheckResult(
        f"Cash-flow gap within {rules.NDCF_CF_GAP_MAX_PCT:g}% (house rule)",
        Status.REVIEW if (not pd.isna(gap) and not bool(gap)) else status.from_bool(gap),
        f"{period}: " + ("cash-flow figures missing" if pd.isna(gap) else f"gap {row['Gap % of Computed']:.1f}% of computed NDCF"),
        AREA, "ndcf.cf_gap_max")
    return [payout, timeline, gap_result]


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
        good_payout = int(q["Meets payout rule"].astype("boolean").fillna(False).sum())
        good_gap = int(q["Within gap limit"].astype("boolean").fillna(False).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric(f"TRUST: periods meeting {rules.NDCF_PAYOUT_MIN_PCT:g}% payout", f"{good_payout}/{total}")
        c2.metric(f"TRUST: periods within {rules.NDCF_CF_GAP_MAX_PCT:g}% gap", f"{good_gap}/{total}")
        c3.metric("TRUST: rows analysed", f"{total}")
        n_unknown_payout = int(q["Meets payout rule"].isna().sum())
        n_unknown_gap = int(q["Within gap limit"].isna().sum())
        if n_unknown_payout or n_unknown_gap:
            st.caption(
                f"Insufficient data (shown as {status.NO_DATA_TAG}): {n_unknown_payout} period(s) for the payout check and "
                f"{n_unknown_gap} for the cash-flow gap check. They are not counted as passes or failures."
            )

        st.subheader(f"Trust Check 1 — {rules.NDCF_PAYOUT_MIN_PCT:g}% payout of Computed NDCF (period-wise)")
        st.caption("SEBI REIT Regulations, Reg. 18(16)(b): not less than 90% of net distributable cash flows to unitholders.")
        disp1 = q[
            [
                "Financial Year",
                "Period Ended",
                "Total Amount of NDCF computed as per NDCF Statement",
                "Total Amount of NDCF declared for the period (incl. Surplus)",
                "Payout Ratio %",
                "Meets payout rule",
            ]
        ].copy()
        disp1["Meets payout rule"] = disp1["Meets payout rule"].map(_status)
        st.dataframe(disp1, width="stretch", hide_index=True)
        if (~q["Meets payout rule"].astype("boolean").fillna(True)).any():
            st.error(
                f"TRUST: One or more periods do **not** meet the {rules.NDCF_PAYOUT_MIN_PCT:g}% payout requirement "
                f"(Declared incl. surplus < {rules.NDCF_PAYOUT_MIN_PCT:g}% of Computed NDCF)."
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
                "Within gap limit",
            ]
        ].copy()
        disp2["Within gap limit"] = disp2["Within gap limit"].map(_status)
        st.dataframe(disp2, width="stretch", hide_index=True)
        if (~q["Within gap limit"].astype("boolean").fillna(True)).any():
            st.error(f"TRUST: One or more periods have a gap **> {rules.NDCF_CF_GAP_MAX_PCT:g}%** between (CFO + CFI + CFF + PAT) and Computed NDCF.")

        # -------- Split timeline checks into two separate tables ----------
        tline = compute_trust_timeline_checks(q)
        if tline.empty:
            st.info("Declaration / Record / Distribution columns not found; timeline checks skipped.")
        else:
            date_problems = tline[tline["Date check"] != ""]
            if not date_problems.empty:
                st.warning(
                    f"{len(date_problems)} period(s) have missing or impossible dates in the sheet (for example a distribution date before "
                    "the declaration date, often day and month typed the wrong way round). They show as No data in the checks below "
                    "and are **not** counted as late payments. Please correct the dates in the sheet.",
                    icon=":material/warning:",
                )
                with st.expander("Periods with date problems"):
                    st.dataframe(
                        date_problems[["Financial Year", "Period Ended", "Declaration Date", "Record Date", "Distribution Date", "Date check"]],
                        width="stretch", hide_index=True,
                    )

            st.caption(
                f"Distribution timeline (REIT Regulations, Reg. 18(16)(c)): declared before {rules.NDCF_NEW_TIMELINE_FROM:%d %b %Y}, "
                f"payment within {rules.NDCF_DISTRIBUTION_MAX_DAYS} days of declaration; declared on or after that date, record date within "
                f"{rules.NDCF_RECORD_MAX_WORKING_DAYS} working days of declaration and payment within "
                f"{rules.NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS} working days of the record date. Each period is checked against the "
                "rule for its declaration date; a check that does not apply shows n/a. Working days are Monday to Friday (market "
                "holidays are not excluded)."
            )

            def _shown(check_col):
                """Status text, with n/a where the check belongs to the other timeline (a date problem shows —)."""
                applicable = tline["Rule applied"].eq(RULE_NEW) if check_col != "Distribution within limit" else tline["Rule applied"].eq(RULE_OLD)
                return tline[check_col].map(_status).where(~(tline[check_col].isna() & tline["Rule applied"].ne("") & ~applicable), status.NA_TAG)

            st.subheader(f"Trust Check 3 — Declaration → Record Date (≤ {rules.NDCF_RECORD_MAX_WORKING_DAYS} working days, from {rules.NDCF_NEW_TIMELINE_FROM:%d %b %Y})")
            t1 = tline[
                ["Financial Year", "Period Ended", "Declaration Date", "Record Date", "Rule applied", "Days Decl→Record", "Working days Decl→Record", "Record on time", "Date check"]
            ].copy()
            t1["Record on time"] = _shown("Record on time")
            st.dataframe(t1, width="stretch", hide_index=True)
            if (tline["Record on time"] == False).any():
                st.error(f"TRUST: One or more periods have **Record Date more than {rules.NDCF_RECORD_MAX_WORKING_DAYS} working days after Declaration**.")

            st.subheader(f"Trust Check 4 — Record Date → Distribution Date (≤ {rules.NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS} working days, from {rules.NDCF_NEW_TIMELINE_FROM:%d %b %Y})")
            t2 = tline[
                ["Financial Year", "Period Ended", "Record Date", "Distribution Date", "Rule applied", "Days Record→Distr", "Working days Record→Distr", "Distribution on time", "Date check"]
            ].copy()
            t2["Distribution on time"] = _shown("Distribution on time")
            st.dataframe(t2, width="stretch", hide_index=True)
            if (tline["Distribution on time"] == False).any():
                st.error(f"TRUST: One or more periods have **Distribution Date more than {rules.NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS} working days after Record Date**.")

            st.subheader(f"Trust Check 5 — Declaration → Distribution Date (≤ {rules.NDCF_DISTRIBUTION_MAX_DAYS} days, declared before {rules.NDCF_NEW_TIMELINE_FROM:%d %b %Y})")
            t3 = tline[
                ["Financial Year", "Period Ended", "Declaration Date", "Distribution Date", "Rule applied", "Days Decl→Distr", "Distribution within limit", "Date check"]
            ].copy()
            t3["Distribution within limit"] = _shown("Distribution within limit")
            st.dataframe(t3, width="stretch", hide_index=True)
            if (tline["Distribution within limit"] == False).any():
                st.error(
                    f"TRUST: One or more periods have **Distribution Date more than {rules.NDCF_DISTRIBUTION_MAX_DAYS} days after Declaration** "
                    "(REIT Regulations, Reg. 18(16)(c); late payment also carries interest at 15% p.a., which is not calculated here)."
                )
            checked = tline[["Record on time", "Distribution on time", "Distribution within limit"]]
            if (tline["Rule applied"].ne("") & checked.isna().all(axis=1)).any():
                st.info("Periods shown as No data have missing or out-of-order dates, so the timeline could not be checked for them.")

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

        st.subheader(f"SPV Check 1 — Declared (incl. Surplus) ≥ {rules.NDCF_PAYOUT_MIN_PCT:g}% of Computed NDCF (by SPV/period)")
        disp_s1 = q[
            [
                "Name of SPV",
                "Name of Holdco (Leave Blank if N/A)",
                "Financial Year",
                "Period Ended",
                "Total Amount of NDCF computed as per NDCF Statement",
                "Total Amount of NDCF declared for the period (incl. Surplus)",
                "Payout Ratio %",
                "Meets payout rule (SPV)",
            ]
        ].copy()
        disp_s1["Meets payout rule (SPV)"] = disp_s1["Meets payout rule (SPV)"].map(_status)
        st.dataframe(disp_s1, width="stretch", hide_index=True)
        if (~q["Meets payout rule (SPV)"].astype("boolean").fillna(True)).any():
            st.error(f"SPV: One or more SPV periods do **not** meet the {rules.NDCF_PAYOUT_MIN_PCT:g}% payout requirement.")

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
        st.dataframe(disp_s2, width="stretch", hide_index=True)
        if (~q["Within Computed Bound (SPV)"].astype("boolean").fillna(True)).any():
            st.error("SPV: One or more SPV periods have |Gap| ≥ Computed NDCF.")


# Entry point used by pages/5_NDCF.py
def render_ndcf():
    render()


if __name__ == "__main__":
    render()
