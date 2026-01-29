# tabs/borrowings.py
from wsgiref import types
import pandas as pd
import streamlit as st
from tabs.fundraising import multiselect_with_select_all
from utils.common import (
    ENT_COL, FY_COL, QTR_COL, EPS,
    DEFAULT_REIT_BORR_URL, DEFAULT_INVIT_BORR_URL,
    _to_date, _to_pct, _is_taken, _is_yes, _is_aaa,
    _find_col, _num_series, _standardize_selector_columns, _quarter_sort,
    _url, load_table_url
)

# --- Constants for Business Logic ---
INVIT_NBR_CAP = 0.70
INVIT_AAA_THRESHOLD = 0.49
CREDIT_RATING_THRESHOLD = 0.25
REIT_NBR_THRESHOLD = 0.25

def _process_borrowings_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = _standardize_selector_columns(df)

    cols = df.columns
    borrow_col = _find_col(cols, aliases=["Borrowings","A. Borrowings","A - Borrowings"], must_tokens=["borrow"])
    defer_col  = _find_col(cols, aliases=["Deferred Payments","B. Deferred Payments"], must_tokens=["defer","payment"])
    cash_col   = _find_col(cols, aliases=["Cash and Cash Equivalents","C. Cash and Cash Equivalents"], must_tokens=["cash","equivalent"])
    assets_col = _find_col(cols, aliases=["Value of REIT Assets","D. Value of REIT Assets","Value of InvIT Assets"], must_tokens=["value","asset"])

    A = _num_series(df, borrow_col, 0.0)
    B = _num_series(df, defer_col, 0.0)
    C = _num_series(df, cash_col, 0.0)
    D = _num_series(df, assets_col, pd.NA).replace(0, pd.NA)

    nbr_col = _find_col(cols, aliases=["Net Borrowings Ratio (NBR)"], must_tokens=["borrow","ratio","nbr"])
    if nbr_col:
        df["NBR_ratio"] = df[nbr_col].apply(_to_pct)

    if "NBR_ratio" not in df.columns or df["NBR_ratio"].isna().any():
        computed = (A.add(B, fill_value=0).sub(C, fill_value=0)) / D
        df["NBR_ratio"] = df.get("NBR_ratio", computed).fillna(computed)

    for col in ["Date of Publishing Credit Rating CRA1", "Date of Publishing Credit Rating CRA2", "Date of meeting for Unitholder Approval", "Date Of intimation to Trustee"]:
        if col in df.columns:
            df[f"{col} (fmt)"] = df[col].apply(_to_date)

    df.attrs["__matched_cols__"] = {
        "Borrowings": borrow_col, "Deferred Payments": defer_col,
        "Cash and Cash Equivalents": cash_col, "Value of REIT/Trust Assets": assets_col,
        "NBR source": nbr_col or "computed",
    }
    return df

@st.cache_data(show_spinner=False, ttl=300)
def load_borrowings_url(url: str) -> pd.DataFrame:
    df = load_table_url(url)
    return _process_borrowings_df(df)

def _render_card_breakup(row, m):
    a_label = m.get("Borrowings") or "Borrowings"
    b_label = m.get("Deferred Payments") or "Deferred Payments"
    c_label = m.get("Cash and Cash Equivalents") or "Cash and Cash Equivalents"
    d_label = m.get("Value of REIT/Trust Assets") or "Value of REIT Assets"
    st.markdown(f"""
        **Breakup**
        - **A. Borrowings**: {row.get(a_label, "-")}
        - **B. Deferred Payments**: {row.get(b_label, "-")}
        - **C. Cash and Cash Equivalents**: {row.get(c_label, "-")}
        - **D. Value of REIT Assets**: {row.get(d_label, "-")}
    """)

def _alerts_and_sections(row, ruleset: str):
    nbr = row.get("NBR_ratio", None)
    if not isinstance(nbr, (int, float)) or pd.isna(nbr):
        st.info("NBR not available. Compliance sections cannot be displayed.")
        return False

    if ruleset == "invit":
        if nbr > INVIT_NBR_CAP + EPS:
            st.error(f"ALERT: NBR is {float(nbr)*100:.2f}% which exceeds the {INVIT_NBR_CAP*100:.0f}% cap for InvITs.")
        show_sections = (nbr > CREDIT_RATING_THRESHOLD + EPS)
    else: # reit
        show_sections = (nbr >= REIT_NBR_THRESHOLD - EPS)

    if not show_sections:
        st.info("NBR is below the threshold. Credit Rating and Unitholder Approval sections are not required.")
        return False
    return True

def _check_compliance_alerts(row, ruleset: str):
    cols = row.index
    cra1_rating = row.get(_find_col(cols, aliases=["Credit Rating CRA1"]))
    cra2_rating = row.get(_find_col(cols, aliases=["Credit Rating CRA2"]))
    ua_col = _find_col(cols, must_tokens=["unitholder", "approval"], exclude_tokens=["date", "meeting", "weblink"])
    unitholder_approval_val = row.get(ua_col)

    credit_taken_any = _is_taken(cra1_rating) or _is_taken(cra2_rating)
    aaa_ok = _is_aaa(cra1_rating) or _is_aaa(cra2_rating)
    unit_taken = _is_yes(unitholder_approval_val)

    missing = []
    nbr = row.get("NBR_ratio", 0.0) or 0.0
    if ruleset == "invit":
        if nbr > INVIT_AAA_THRESHOLD + EPS:
            if not aaa_ok: missing.append("AAA Credit Rating")
            if not unit_taken: missing.append("Unitholder Approval")
        elif nbr > CREDIT_RATING_THRESHOLD + EPS:
            if not credit_taken_any: missing.append("Credit Rating")
            if not unit_taken: missing.append("Unitholder Approval")
    else: # reit
        if not credit_taken_any: missing.append("Credit Rating")
        if not unit_taken: missing.append("Unitholder Approval")

    if missing:
        msg = f"Both {missing[0]} and {missing[1]} are not taken/available." if len(missing) == 2 else f"{missing[0]} is not taken/available."
        st.error(f"ALERT: {msg} for this period.")

def _render_credit_rating_ui(row):
    cols = row.index
    st.markdown("### Credit Rating")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**CRA1**")
        rating = row.get(_find_col(cols, aliases=["Credit Rating CRA1"]))
        st.write(f"**Rating**: {rating if _is_taken(rating) else '-'}")
        st.write(f"**Name**: {row.get(_find_col(cols, aliases=['Name of CRA1']), '-')}")
        st.write(f"**Date**: {row.get(_find_col(cols, aliases=['Date of Publishing Credit Rating CRA1 (fmt)']), '-')}")
        link = _url(row.get(_find_col(cols, aliases=['Weblink of CRA1 Disclosure (CRA/Exchange)'])))
        if link: st.markdown(f"**Disclosure Link**: [View Document]({link})")
    with c2:
        st.markdown("**CRA2**")
        rating = row.get(_find_col(cols, aliases=["Credit Rating CRA2"]))
        st.write(f"**Rating**: {rating if _is_taken(rating) else '-'}")
        st.write(f"**Name**: {row.get(_find_col(cols, aliases=['Name of CRA2']), '-')}")
        st.write(f"**Date**: {row.get(_find_col(cols, aliases=['Date of Publishing Credit Rating CRA2 (fmt)']), '-')}")
        link = _url(row.get(_find_col(cols, aliases=['Weblink of CRA2 Disclosure (CRA/Exchange)'])))
        if link: st.markdown(f"**Disclosure Link**: [View Document]({link})")
    st.markdown("---")


def _render_unitholder_and_compliances_ui(row):
    cols = row.index
    st.markdown("### Unitholder Approval & Compliances")

    # Unitholder Approval part
    ua_col = _find_col(cols, must_tokens=["unitholder", "approval"], exclude_tokens=["date", "meeting", "weblink"])
    ua_val = row.get(ua_col)
    approval_display = "Yes" if _is_yes(ua_val) else ("No" if _is_taken(ua_val) else "-")
    st.write(f"**Unitholder Approval Taken**: {approval_display}")
    st.write(f"**Date of meeting**: {row.get('Date of meeting for Unitholder Approval (fmt)', '-')}")

    # ** THIS IS THE CORRECTED LINK LOGIC **
    link_col = _find_col(cols, must_tokens=["weblink", "unitholder"])
    link = _url(row.get(link_col))
    if link:
        st.markdown(f"**Disclosure Link**: [View Document]({link})")

    # Additional Compliances part
    st.write(f"**Whether NBR > 25% due to market movement?** {row.get('Whether NBR>25% on account of market movement?', '-')}")
    st.write(f"**Date of intimation to Trustee**: {row.get('Date Of intimation to Trustee (fmt)', '-')}")


def render():
    st.header("Borrowings")
    with st.sidebar:
        segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_borrow")
        # compute the default URL after segment is chosen
        
        data_url = DEFAULT_INVIT_BORR_URL if segment == "InvIT" else DEFAULT_REIT_BORR_URL
        
    if not data_url.strip():
        st.warning("Please provide a data URL."); st.stop()

    try:
        df = load_borrowings_url(data_url.strip())
    except Exception as e:
        st.error(f"Could not read the URL. Make sure it’s publicly accessible.\n\nDetails: {e}"); st.stop()

    with st.sidebar:
        st.divider()
        entity = multiselect_with_select_all(
            "Entity", sorted(df[ENT_COL].dropna().astype(str).unique()), key=f"entity_{segment}", default_all=False,
            help="Use “Select all” at the top to include every entity.",
        )
        fy = multiselect_with_select_all("Financial Year", sorted(df.loc[df[ENT_COL] == entity, FY_COL].dropna().astype(str).unique()), key=f"fy_{segment}", default_all=False)
        qtr_present = df.loc[(df[ENT_COL] == entity) & (df[FY_COL] == fy), QTR_COL].dropna().astype(str).unique().tolist()
        qtr = multiselect_with_select_all("Quarter", _quarter_sort(qtr_present), key=f"qtr_{segment}", default_all=False)
    
    row_df = df[(df[ENT_COL] == entity) & (df[FY_COL] == fy) & (df[QTR_COL] == qtr)]
    if row_df.empty: st.warning("No data found for the selected filters."); st.stop()
    row = row_df.iloc[0]

    # KPI + Breakup
    colA, colB = st.columns([0.9, 1.1])
    with colA:
        nbr = row.get("NBR_ratio", None)
        nbr_display = "-" if pd.isna(nbr) else f"{float(nbr)*100:.2f}%"
        st.markdown(f'<div class="kpi">📊 <b>Net Borrowings Ratio</b><br><span class="kpi-value">{nbr_display}</span></div>', unsafe_allow_html=True)
        if isinstance(nbr, (int, float)) and not pd.isna(nbr): st.progress(min(max(float(nbr), 0.0), 1.0))
    with colB:
        _render_card_breakup(row, df.attrs.get("__matched_cols__", {}))

    st.markdown("---")

    ruleset = "invit" if segment == "InvIT" else "reit"
    if _alerts_and_sections(row, ruleset):
        _check_compliance_alerts(row, ruleset)
        _render_credit_rating_ui(row)
        _render_unitholder_and_compliances_ui(row)