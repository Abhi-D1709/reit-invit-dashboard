# tabs/borrowings.py
import pandas as pd
import streamlit as st
from utils.common import (
    ENT_COL, FY_COL, QTR_COL, EPS, AAA_PAT,
    DEFAULT_REIT_BORR_URL, DEFAULT_INVIT_BORR_URL,
    _to_date, _to_pct, _to_num, _is_taken, _is_yes, _is_aaa,
    _find_col, _num_series, _standardize_selector_columns, _quarter_sort,
    _url, load_table_url
)

def _process_borrowings_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = _standardize_selector_columns(df)

    cols = df.columns
    borrow_col = _find_col(cols, aliases=["Borrowings","A. Borrowings","A - Borrowings","A Borrowings"], must_tokens=["borrow"])
    defer_col  = _find_col(cols, aliases=["Deferred Payments","B. Deferred Payments","Deferred Payment"], must_tokens=["defer","payment"])
    cash_col   = _find_col(cols, aliases=["Cash and Cash Equivalents","C. Cash and Cash Equivalents","Cash & Cash Equivalents","Cash and cash equivalents"], must_tokens=["cash","equivalent"])
    assets_col = _find_col(cols, aliases=["Value of REIT Assets","D. Value of REIT Assets","Value of Assets","Value of InvIT Assets","Value of Trust Assets"], must_tokens=["value","asset"])

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

    for col in [
        "Date of Publishing Credit Rating CRA1",
        "Date of Publishing Credit Rating CRA2",
        "Date of meeting for Unitholder Approval",
        "Date Of intimation to Trustee",
    ]:
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
    st.markdown("**Breakup**")
    st.write(
        f"""
- **A. Borrowings**: {row.get(a_label, "-")}
- **B. Deferred Payments**: {row.get(b_label, "-")}
- **C. Cash and Cash Equivalents**: {row.get(c_label, "-")}
- **D. Value of REIT Assets**: {row.get(d_label, "-")}
"""
    )

def _alerts_and_sections(row, ruleset: str):
    nbr = row.get("NBR_ratio", None)
    # InvIT rules
    if ruleset == "invit":
        if isinstance(nbr, (int, float)) and not pd.isna(nbr) and (nbr > 0.70 + EPS):
            st.error(f"ALERT: NBR is {float(nbr)*100:.2f}% which exceeds the 70% cap for InvITs.")
        show_sections = isinstance(nbr, (int, float)) and not pd.isna(nbr) and (nbr > 0.25 + EPS)
    else:
        show_sections = isinstance(nbr, (int, float)) and not pd.isna(nbr) and (nbr >= 0.25 - EPS)

    if not show_sections:
        st.info("NBR is below the threshold. Credit Rating, Unitholder Approval, and Additional Compliances are not required to be displayed.")
        return False
    return True

def _credit_and_ua_blocks(row, ruleset: str):
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

    cra1_rating = row.get(cra1_rating_col); cra2_rating = row.get(cra2_rating_col)
    cra1_name   = row.get(cra1_name_col);   cra2_name   = row.get(cra2_name_col)
    cra1_date   = row.get(cra1_date_col);   cra2_date   = row.get(cra2_date_col)
    cra1_link   = _url(row.get(cra1_link_col)); cra2_link = _url(row.get(cra2_link_col))

    credit_taken_any = _is_taken(cra1_rating) or _is_taken(cra2_rating)
    aaa_ok = _is_aaa(cra1_rating) or _is_aaa(cra2_rating)

    ua_col = _find_col(
        cols,
        aliases=["Unitholder Approval", "Unitholder approval"],
        must_tokens=["unitholder", "approval"],
        exclude_tokens=["date", "meeting", "weblink", "notice", "votes", "record", "favour", "against", "total"]
    )
    unitholder_approval_val = row.get(ua_col)
    unit_taken = _is_yes(unitholder_approval_val)

    # Alerts per rules
    missing = []
    if ruleset == "invit":
        nbr = row.get("NBR_ratio", 0.0) or 0.0
        if nbr > 0.49 + EPS:
            if not aaa_ok:     missing.append("AAA Credit Rating")
            if not unit_taken: missing.append("Unitholder Approval")
        elif nbr > 0.25 + EPS:
            if not credit_taken_any: missing.append("Credit Rating")
            if not unit_taken:       missing.append("Unitholder Approval")
    else:
        if not credit_taken_any: missing.append("Credit Rating")
        if not unit_taken:       missing.append("Unitholder Approval")

    if missing:
        msg = (
            f"Both {missing[0]} and {missing[1]} are not taken / not available for this period."
            if len(missing) == 2 else
            f"{missing[0]} is not taken / not available for this period."
        )
        st.error(f"ALERT: {msg}")

    # Credit Rating block
    st.markdown("### Credit Rating")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**CRA1**")
        st.write(f"**Rating**: {cra1_rating if _is_taken(cra1_rating) else '-'}")
        st.write(f"**Name of CRA**: {cra1_name if _is_taken(cra1_name) else '-'}")
        st.write(f"**Date of Publishing Credit Rating**: {_to_date(cra1_date) if _is_taken(cra1_date) else '-'}")
        if cra1_link: st.markdown(f"[CRA1 Disclosure Link]({cra1_link})")
    with col2:
        st.markdown("**CRA2**")
        st.write(f"**Rating**: {cra2_rating if _is_taken(cra2_rating) else '-'}")
        st.write(f"**Name of CRA**: {cra2_name if _is_taken(cra2_name) else '-'}")
        st.write(f"**Date of Publishing Credit Rating**: {_to_date(cra2_date) if _is_taken(cra2_date) else '-'}")
        if cra2_link: st.markdown(f"[CRA2 Disclosure Link]({cra2_link})")

    updown2 = next((row.get(c) for c in [
        "No. of Rating Upgrades/Downgrades CRA1",
        "No. of Rating Upgrades/Downgrades CRA2",
        "No. of Rating Upgrades/Downgrades"
    ] if c in cols), "-")
    st.write(f"**No. of Rating Upgrades/Downgrades**: {updown2}")

    st.markdown("---")

    # Unitholder Approval block
    st.markdown("### Unitholder Approval")
    approval_display = "Yes" if _is_yes(unitholder_approval_val) else ("No" if _is_taken(unitholder_approval_val) else "-")
    st.write(f"**Approval Taken**: {approval_display}")
    st.write(f"**Date of meeting for Unitholder Approval**: {row.get('Date of meeting for Unitholder Approval (fmt)', '-')}")
    link_um = _url(next((row.get(c) for c in [
        "Weblink of Disclosure of Outcome of Unitholder Meeting (Exchange)",
        "Weblink of Disclosure of Notice for Unitholder Meeting (Exchange)"
    ] if c in cols), "-"))
    if link_um:
        st.markdown(f"[Disclosure on Exchange]({link_um})")
    st.write(f"**Total No. of Unitholders on record date**: {row.get('Total No. of Unitholders on record date', '-')}")
    st.write(f"**Total No. of Votes Cast**: {row.get('Total No. of Votes Cast', '-')}")
    st.write(f"**Votes Cast (Favour/Against)**: {row.get('Votes Cast in Favour/Votes Cast Against', '-')}")

    st.markdown("---")

    # Additional Compliances
    st.markdown("### Additional Compliances")
    st.write(f"**Whether NBR > 25% due to market movement?**  {row.get('Whether NBR>25% on account of market movement?', '-')}")
    st.write(f"**Date of intimation to Trustee**: {row.get('Date Of intimation to Trustee (fmt)', '-')}")

def render():
    st.header("Borrowings")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_borrow")
    default_url = DEFAULT_INVIT_BORR_URL if segment == "InvIT" else DEFAULT_REIT_BORR_URL

    st.subheader("Data Source")
    st.caption("Paste a public URL (Google Sheet / CSV / XLSX / JSON / HTML table).")
    data_url = st.text_input("Data URL", value=default_url, key=f"borr_url_{segment}")

    if not data_url.strip():
        st.warning("Please provide a data URL.")
        st.stop()

    try:
        df = load_borrowings_url(data_url.strip())
    except Exception as e:
        st.error(f"Could not read the URL. Make sure it’s publicly accessible.\n\nDetails: {e}")
        st.stop()

    # Filters
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            entities = sorted(df[ENT_COL].dropna().astype(str).unique())
            entity = st.selectbox("Entity", entities, key=f"entity_{segment}")
        with c2:
            fy_options = sorted(df.loc[df[ENT_COL] == entity, FY_COL].dropna().astype(str).unique())
            fy = st.selectbox("Financial Year", fy_options, key=f"fy_{segment}")
        with c3:
            qtr_present = df.loc[(df[ENT_COL] == entity) & (df[FY_COL] == fy), QTR_COL].dropna().astype(str).unique().tolist()
            qtr = st.selectbox("Quarter", _quarter_sort(qtr_present), key=f"qtr_{segment}")

    row_df = df[(df[ENT_COL] == entity) & (df[FY_COL] == fy) & (df[QTR_COL] == qtr)]
    if row_df.empty:
        st.warning("No data found for the selected filters.")
        st.stop()
    row = row_df.iloc[0]

    # KPI + Breakup
    colA, colB = st.columns([0.9, 1.1])
    with colA:
        nbr = row.get("NBR_ratio", None)
        nbr_display = "-" if nbr is None or pd.isna(nbr) else f"{float(nbr)*100:.2f}%"
        st.markdown(
            '<div class="kpi">📊 <b>Net Borrowings Ratio</b><br>'
            f'<span style="font-size:28px;font-weight:700;">{nbr_display}</span></div>',
            unsafe_allow_html=True
        )
        if isinstance(nbr, (int, float)) and not pd.isna(nbr):
            st.progress(min(max(float(nbr), 0.0), 1.0))
    with colB:
        _render_card_breakup(row, df.attrs.get("__matched_cols__", {}))

    st.markdown("---")

    ruleset = "invit" if segment == "InvIT" else "reit"
    if _alerts_and_sections(row, ruleset):
        _credit_and_ua_blocks(row, ruleset)

    st.markdown("---")
    st.markdown(
        """
        **Common Links**
        - [Indiabondinfo](https://www.indiabondinfo.nsdl.com/)
        - [CDSL BondInfo](https://www.cdslindia.com/CorporateBond/SearchISIN.aspx)
        - [NSE India](https://www.nseindia.com)
        - [BSE India](https://www.bseindia.com)
        """,
    )
