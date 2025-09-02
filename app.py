import os
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="REIT / InvIT Dashboard", page_icon="📊", layout="wide")

DEFAULT_XLSX = "Borrowings.xlsx"  # current dataset: REIT Borrowings only

# Canonical names for selector fields (we map your sheet’s headers to these)
ENT_COL = "__Entity__"
FY_COL  = "__FinancialYear__"
QTR_COL = "__QuarterEnded__"

# ---------- Helpers ----------
def _to_date(val):
    if val is None or (isinstance(val, str) and val.strip() in {"", "-"}) or pd.isna(val):
        return "-"
    try:
        if isinstance(val, (int, float)) and not math.isnan(val):
            base = datetime(1899, 12, 30)  # Excel serial origin
            return (base + timedelta(days=float(val))).date().isoformat()
    except Exception:
        pass
    if isinstance(val, (pd.Timestamp, datetime)):
        return pd.to_datetime(val).date().isoformat()
    dt = pd.to_datetime(str(val), errors="coerce", dayfirst=True)
    return dt.date().isoformat() if not pd.isna(dt) else str(val)

def _to_pct(val):
    if val is None or (isinstance(val, str) and val.strip() == "") or pd.isna(val):
        return None
    if isinstance(val, str) and val.strip().endswith("%"):
        try:
            return float(val.strip().replace("%", "")) / 100.0
        except Exception:
            return None
    try:
        v = float(val)
        return v / 100.0 if v > 1 else v
    except Exception:
        return None

def _is_taken(value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s not in {"", "-", "—", "na", "n/a", "not rated", "no", "none", "null", "nan", "nr", "not applicable", "n.a.", "nil"}

YES_PAT = re.compile(r"\b(yes|y|true|approved|taken)\b|^1$|✓", re.I)
def _is_yes(value):
    if value is None or pd.isna(value):
        return False
    return YES_PAT.search(str(value).strip()) is not None

def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _find_col(columns, aliases=None, must_tokens=None, exclude_tokens=None):
    """Flexible finder: exact alias (case/space-insensitive) or token match."""
    aliases = aliases or []
    must_tokens = [t.replace(" ", "") for t in (must_tokens or [])]
    exclude_tokens = [t.replace(" ", "") for t in (exclude_tokens or [])]
    norm_map = {c: _norm(c) for c in columns}
    norm_aliases = {_norm(a): a for a in aliases}
    for c, n in norm_map.items():  # exact alias
        if n in norm_aliases:
            return c
    for c, n in norm_map.items():  # token match
        if all(t in n for t in must_tokens) and not any(x in n for x in exclude_tokens):
            return c
    return None

def _url(val):
    if not _is_taken(val):
        return None
    s = str(val).strip()
    return s if s.startswith(("http://", "https://")) else f"https://{s}"

def _num_series(df: pd.DataFrame, colname: str, fill=0.0) -> pd.Series:
    """Return numeric Series; if col missing, create Series with `fill`."""
    if colname and colname in df.columns:
        return pd.to_numeric(df[colname], errors="coerce")
    fv = np.nan if (fill is pd.NA or fill is None) else fill
    return pd.Series([fv] * len(df), index=df.index, dtype="float64")

def _standardize_selector_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map your sheet’s headers to canonical Entity / Financial Year / Quarter Ended."""
    cols = df.columns
    entity_col = _find_col(cols,
        aliases=["Entity", "Entity Name", "REIT", "REIT Name", "Name of REIT",
                 "Trust", "Trust Name", "Issuer", "Issuer Name", "InvIT / REIT Name"])
    fy_col = _find_col(cols,
        aliases=["Financial Year", "Fin Year", "FY", "Financial Yr", "Year"],
        must_tokens=["financial", "year"], exclude_tokens=["quarter", "qtr"]) \
        or _find_col(cols, aliases=[], must_tokens=["year"], exclude_tokens=["quarter", "qtr"])
    qtr_col = _find_col(cols,
        aliases=["Quarter Ended", "Quarter", "Qtr", "Q/E", "Quarter (Ended)"],
        must_tokens=["quarter"], exclude_tokens=["year"])

    missing = []
    if entity_col is None: missing.append("Entity (e.g., 'Entity' / 'REIT Name')")
    if fy_col is None:     missing.append("Financial Year (e.g., 'Financial Year' / 'FY')")
    if qtr_col is None:    missing.append("Quarter Ended (e.g., 'Quarter Ended' / 'Quarter')")

    if missing:
        df.attrs["__fatal__"] = (
            "Missing required selector columns:\n- " + "\n- ".join(missing) +
            "\n\nAvailable columns:\n- " + "\n- ".join(map(str, cols))
        )
        for cname in (ENT_COL, FY_COL, QTR_COL):
            if cname not in df.columns:
                df[cname] = np.nan
        return df

    df = df.rename(columns={entity_col: ENT_COL, fy_col: FY_COL, qtr_col: QTR_COL})
    df.attrs["__selector_map__"] = {"Entity": entity_col, "Financial Year": fy_col, "Quarter Ended": qtr_col}
    return df

def _quarter_sort(values):
    order = {"June": 0, "Sept": 1, "Sep": 1, "December": 2, "Dec": 2, "Mar": 3, "March": 3}
    return sorted(values, key=lambda v: order.get(str(v), 99))

# ---------- Data loader for Borrowings (REIT) ----------
@st.cache_data(show_spinner=False, ttl=120)
def load_borrowings(uploaded_file):
    # Read file (uploaded or local)
    df = pd.read_excel(uploaded_file if uploaded_file is not None else DEFAULT_XLSX, sheet_name=0)

    # Normalize headers
    df.columns = [c.strip() for c in df.columns]

    # Standardize selectors
    df = _standardize_selector_columns(df)

    # Numeric column detection
    cols = df.columns
    borrow_col = _find_col(cols, aliases=["Borrowings", "A. Borrowings", "A - Borrowings", "A Borrowings"], must_tokens=["borrow"])
    defer_col  = _find_col(cols, aliases=["Deferred Payments", "B. Deferred Payments", "Deferred Payment"], must_tokens=["defer", "payment"])
    cash_col   = _find_col(cols, aliases=["Cash and Cash Equivalents", "C. Cash and Cash Equivalents", "Cash & Cash Equivalents", "Cash and cash equivalents"], must_tokens=["cash", "equivalent"])
    assets_col = _find_col(cols, aliases=["Value of REIT Assets", "D. Value of REIT Assets", "Value of Assets"], must_tokens=["value", "asset", "reit"])

    # Series (never scalars)
    A = _num_series(df, borrow_col, 0.0)
    B = _num_series(df, defer_col, 0.0)
    C = _num_series(df, cash_col, 0.0)
    D = _num_series(df, assets_col, np.nan)

    # NBR normalize / compute; guard D==0
    nbr_col = _find_col(cols, aliases=["Net Borrowings Ratio (NBR)"], must_tokens=["borrow", "ratio", "nbr"])
    if nbr_col:
        df["NBR_ratio"] = df[nbr_col].apply(_to_pct)
    if "NBR_ratio" not in df.columns or df["NBR_ratio"].isna().any():
        D_safe = D.replace(0, np.nan)
        computed = (A.add(B, fill_value=0).sub(C, fill_value=0)) / D_safe
        df["NBR_ratio"] = df.get("NBR_ratio", computed).fillna(computed)

    # Date formatting
    for col in [
        "Date of Publishing Credit Rating CRA1",
        "Date of Publishing Credit Rating CRA2",
        "Date of meeting for Unitholder Approval",
        "Date Of intimation to Trustee",
    ]:
        if col in df.columns:
            df[f"{col} (fmt)"] = df[col].apply(_to_date)

    # Debug info
    df.attrs["__matched_cols__"] = {
        "Borrowings": borrow_col, "Deferred Payments": defer_col,
        "Cash and Cash Equivalents": cash_col, "Value of REIT Assets": assets_col,
        "NBR source": nbr_col or "computed",
    }
    return df

# ---------- UI ----------
st.title("REIT / InvIT Dashboard")

tab_fund, tab_borrow, tab_ndcf = st.tabs(["Fund Raising", "Borrowings", "NDCF"])

# ---------------- Fund Raising ----------------
with tab_fund:
    st.header("Fund Raising")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_fund")
    st.info(f"{segment} Fund Raising dashboard will appear here once data is available.")

# ---------------- Borrowings ----------------
with tab_borrow:
    st.header("Borrowings")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_borrow")

    if segment == "InvIT":
        st.info("InvIT Borrowings dashboard will appear here once data is available.")
    else:
        # Data controls (kept inside this tab)
        up = st.file_uploader("Upload updated Borrowings.xlsx (optional)", type=["xlsx"])
        df = load_borrowings(up)

        fatal = df.attrs.get("__fatal__")
        if fatal:
            st.error(fatal)
            st.stop()

        # Filters
        filt1, filt2, filt3 = st.columns(3)
        with filt1:
            entities = sorted(df[ENT_COL].dropna().astype(str).unique())
            entity = st.selectbox("Entity", entities)
        with filt2:
            fy_options = sorted(df.loc[df[ENT_COL] == entity, FY_COL].dropna().astype(str).unique())
            fy = st.selectbox("Financial Year", fy_options)
        with filt3:
            qtr_present = df.loc[(df[ENT_COL] == entity) & (df[FY_COL] == fy), QTR_COL].dropna().astype(str).unique().tolist()
            qtr = st.selectbox("Quarter", _quarter_sort(qtr_present))

        # Row selection
        row_df = df[(df[ENT_COL] == entity) & (df[FY_COL] == fy) & (df[QTR_COL] == qtr)]
        if row_df.empty:
            st.warning("No data found for the selected filters.")
            st.stop()
        row = row_df.iloc[0]

        st.divider()

        # ---------- NBR + components ----------
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Net Borrowings Ratio (NBR) = (A+B−C)/D")
            nbr = row.get("NBR_ratio", None)
            nbr_display = "-" if nbr is None or pd.isna(nbr) else f"{float(nbr)*100:.2f}%"
            st.metric("NBR", nbr_display)
            if isinstance(nbr, (int, float)) and not pd.isna(nbr):
                st.progress(min(max(float(nbr), 0.0), 1.0))

        with right:
            st.subheader("Breakup")
            m = df.attrs.get("__matched_cols__", {})
            a_label = m.get("Borrowings") or "Borrowings"
            b_label = m.get("Deferred Payments") or "Deferred Payments"
            c_label = m.get("Cash and Cash Equivalents") or "Cash and Cash Equivalents"
            d_label = m.get("Value of REIT Assets") or "Value of REIT Assets"
            st.write(
                f"""
- **A. Borrowings**: {row.get(a_label, "-")}
- **B. Deferred Payments**: {row.get(b_label, "-")}
- **C. Cash and Cash Equivalents**: {row.get(c_label, "-")}
- **D. Value of REIT Assets**: {row.get(d_label, "-")}
"""
            )

        # ---------- Conditional sections ----------
        SHOW_THRESHOLD = 0.25
        show_sections = isinstance(nbr, (int, float)) and not pd.isna(nbr) and nbr >= SHOW_THRESHOLD

        if not show_sections:
            st.info("NBR is below 25%. Credit Rating, Unitholder Approval, and Additional Compliances are not required to be displayed.")
        else:
            cols = row.index

            # CRA1 / CRA2
            cra1_rating_col = _find_col(cols, aliases=["Credit Rating CRA1"])
            cra2_rating_col = _find_col(cols, aliases=["Credit Rating CRA2"])
            cra1_name_col   = _find_col(cols, aliases=["Name of CRA1"])
            cra2_name_col   = _find_col(cols, aliases=["Name of CRA2"])
            cra1_date_col   = _find_col(cols, aliases=["Date of Publishing Credit Rating CRA1 (fmt)", "Date of Publishing Credit Rating CRA1"])
            cra2_date_col   = _find_col(cols, aliases=["Date of Publishing Credit Rating CRA2 (fmt)", "Date of Publishing Credit Rating CRA2"])
            cra1_link_col   = _find_col(cols, aliases=["Weblink of CRA1 Disclosure (CRA/Exchange)"])
            cra2_link_col   = _find_col(cols, aliases=["Weblink of CRA2 Disclosure (CRA/Exchange)"])

            cra1_rating = row.get(cra1_rating_col)
            cra2_rating = row.get(cra2_rating_col)
            cra1_name   = row.get(cra1_name_col)
            cra2_name   = row.get(cra2_name_col)
            cra1_date   = row.get(cra1_date_col)
            cra2_date   = row.get(cra2_date_col)
            cra1_link   = _url(row.get(cra1_link_col))
            cra2_link   = _url(row.get(cra2_link_col))

            credit_taken = _is_taken(cra1_rating) or _is_taken(cra2_rating)

            # Unitholder Approval (correct spelling only)
            ua_col = _find_col(
                cols,
                aliases=["Unitholder Approval", "Unitholder approval"],
                must_tokens=["unitholder", "approval"],
                exclude_tokens=["date", "meeting", "weblink", "notice", "votes", "record", "favour", "against", "total"]
            )
            unitholder_approval_val = row.get(ua_col)
            unit_taken = _is_yes(unitholder_approval_val)

            # Alerts
            missing = []
            if not credit_taken:
                missing.append("Credit Rating")
            if not unit_taken:
                missing.append("Unitholder Approval")
            if missing:
                msg = ("Both Credit Rating and Unitholder Approval are not taken / not available for this period, even though NBR ≥ 25%."
                       if len(missing) == 2 else
                       f"{missing[0]} is not taken / not available for this period, even though NBR ≥ 25%.")
                st.error(f"ALERT: {msg}")

            # ---- Credit Rating (dual) ----
            st.subheader("Credit Rating")
            st.caption("Source: Website of Exchange and/or Website of Entity / CRA website; Annual Report")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**CRA1**")
                st.write(f"**Rating**: {cra1_rating if _is_taken(cra1_rating) else '-'}")
                st.write(f"**Name of CRA**: {cra1_name if _is_taken(cra1_name) else '-'}")
                st.write(f"**Date of Publishing Credit Rating**: {_to_date(cra1_date) if _is_taken(cra1_date) else '-'}")
                if cra1_link: st.markdown(f"[CRA1 Disclosure Link]({cra1_link})")
            with c2:
                st.markdown("**CRA2**")
                st.write(f"**Rating**: {cra2_rating if _is_taken(cra2_rating) else '-'}")
                st.write(f"**Name of CRA**: {cra2_name if _is_taken(cra2_name) else '-'}")
                st.write(f"**Date of Publishing Credit Rating**: {_to_date(cra2_date) if _is_taken(cra2_date) else '-'}")
                if cra2_link: st.markdown(f"[CRA2 Disclosure Link]({cra2_link})")

            # Optional combined metric (if present)
            updown2 = next((row.get(c) for c in [
                "No. of Rating Upgrades/Downgrades CRA1",
                "No. of Rating Upgrades/Downgrades CRA2",
                "No. of Rating Upgrades/Downgrades"
            ] if c in cols), "-")
            st.write(f"**No. of Rating Upgrades/Downgrades**: {updown2}")

            st.divider()

            # ---- Unitholder Approval ----
            st.subheader("Unitholder Approval")
            st.caption("Source: Website of Exchange and/or Website of Entity; Annual Report")
            approval_display = "Yes" if unit_taken else ("No" if _is_taken(unitholder_approval_val) else "-")
            st.write(f"**Approval Taken**: {approval_display}")
            st.write(f"**Date of meeting for Unitholder Approval**: {row.get('Date of meeting for Unitholder Approval (fmt)', '-')}")
            link_um = _url(next((row.get(c) for c in [
                "Weblink of Disclosure of Notice for Unitholder Meeting (Exchange)"
            ] if c in cols), "-"))
            if link_um:
                st.markdown(f"[Notice on Exchange]({link_um})")

            st.write(f"**Total No. of Unitholders on record date**: {row.get('Total No. of Unitholders on record date', '-')}")
            st.write(f"**Total No. of Votes Cast**: {row.get('Total No. of Votes Cast', '-')}")
            st.write(f"**Votes Cast (Favour/Against)**: {row.get('Votes Cast in Favour/Votes Cast Against', '-')}")

            st.divider()

            # ---- Additional Compliances ----
            st.subheader("Additional Compliances")
            st.write(f"**Whether NBR > 25% due to market movement?**  {row.get('Whether NBR>25% on account of market movement?', '-')}")
            st.write(f"**Date of intimation to Trustee**: {row.get('Date Of intimation to Trustee (fmt)', '-')}")

        st.divider()
        st.markdown(
            """
- [Indiabondinfo](https://www.indiabondinfo.nsdl.com/)
- [CDSL BondInfo](https://www.cdslindia.com/CorporateBond/SearchISIN.aspx)
- [NSE India](https://www.nseindia.com)
- [BSE India](https://www.bseindia.com)
"""
        )

# ---------------- NDCF ----------------
with tab_ndcf:
    st.header("NDCF")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_ndcf")
    st.info(f"{segment} NDCF dashboard will appear here once data is available.")
