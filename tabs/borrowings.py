# tabs/borrowings.py
import numpy as np
import pandas as pd
import streamlit as st
from utils.common import (
    ENT_COL, FY_COL, QTR_COL, EPS,
    DEFAULT_REIT_BORR_URL, DEFAULT_INVIT_BORR_URL,
    _to_date, _is_taken, _is_yes, _is_aaa,
    _find_col, _num_series, _standardize_selector_columns, _quarter_sort,
    _url, load_table_url, resolve_percent_units
)
from utils import periods, rules, status  # every threshold lives in utils/rules.py
from utils.status import CheckResult, Status

def _process_borrowings_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = _standardize_selector_columns(df)

    cols = df.columns
    borrow_col = _find_col(cols, aliases=["Borrowings","A. Borrowings","A - Borrowings"], must_tokens=["borrow"])
    defer_col  = _find_col(cols, aliases=["Deferred Payments","B. Deferred Payments"], must_tokens=["defer","payment"])
    cash_col   = _find_col(cols, aliases=["Cash and Cash Equivalents","C. Cash and Cash Equivalents"], must_tokens=["cash","equivalent"])
    assets_col = _find_col(cols, aliases=["Value of REIT Assets","D. Value of REIT Assets","Value of InvIT Assets"], must_tokens=["value","asset"])

    # NBR computed from the components needs Borrowings, Cash and Assets; a blank Deferred Payments
    # means "none". A missing column or cell used to be read as 0, which gave a wrong ratio.
    A = _num_series(df, borrow_col)
    B = _num_series(df, defer_col).fillna(0.0)
    C = _num_series(df, cash_col)
    D = _num_series(df, assets_col).where(lambda s: s != 0)  # zero assets = unknown
    computed = (A + B - C) / D

    nbr_col = _find_col(cols, aliases=["Net Borrowings Ratio (NBR)"], must_tokens=["borrow","ratio","nbr"])
    if nbr_col:
        # The sheet mixes "26.09%", 26.09 and 0.2609, even within one column and per entity, so the
        # unit is decided per cell with the computed ratio as a check (see resolve_percent_units).
        sheet_nbr, how = resolve_percent_units(df[nbr_col], reference=computed, groups=df[ENT_COL])
        df["NBR_ratio"] = sheet_nbr.fillna(computed)
        df["NBR_how"] = how.where(sheet_nbr.notna(), np.where(computed.notna(), "computed from the components", ""))
    else:
        df["NBR_ratio"] = computed
        df["NBR_how"] = np.where(computed.notna(), "computed from the components", "")
    df["NBR_computed"] = computed

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

def nbr_cap(ruleset: str) -> float:
    """The net-borrowings cap for this kind of trust."""
    return rules.INVIT_NBR_CAP if ruleset == "invit" else rules.REIT_NBR_CAP


def nbr_over_cap(nbr: float, ruleset: str) -> bool:
    """Net borrowings above the cap (REIT Regulations, Reg. 20(2) for REITs; InvIT rule as configured)."""
    return nbr > nbr_cap(ruleset) + EPS


def compliance_sections_required(nbr: float, ruleset: str) -> bool:
    """Whether credit rating and unitholder approval apply at this NBR."""
    if ruleset == "invit":
        return nbr > rules.INVIT_RATING_TRIGGER + EPS
    return nbr >= rules.REIT_NBR_TRIGGER - EPS  # REIT Regulations, Reg. 20(3)


def missing_compliance_items(nbr: float, ruleset: str, credit_taken_any: bool, aaa_ok: bool, unit_taken: bool) -> list:
    missing = []
    if ruleset == "invit":
        if nbr > rules.INVIT_AAA_THRESHOLD + EPS:
            if not aaa_ok: missing.append("AAA Credit Rating")
            if not unit_taken: missing.append("Unitholder Approval")
        elif nbr > rules.INVIT_RATING_TRIGGER + EPS:
            if not credit_taken_any: missing.append("Credit Rating")
            if not unit_taken: missing.append("Unitholder Approval")
    else: # reit
        if not credit_taken_any: missing.append("Credit Rating")
        if not unit_taken: missing.append("Unitholder Approval")
    return missing


def _alerts_and_sections(row, ruleset: str):
    nbr = row.get("NBR_ratio", None)
    if not isinstance(nbr, (int, float)) or pd.isna(nbr):
        st.info("NBR not available. Compliance sections cannot be displayed.")
        return False

    if nbr_over_cap(nbr, ruleset):
        kind = "InvITs" if ruleset == "invit" else "REITs"
        msg = f"ALERT: NBR is {float(nbr)*100:.2f}% which exceeds the {nbr_cap(ruleset)*100:.0f}% cap for {kind}."
        if ruleset != "invit":
            msg += (" Under Reg. 20(4) a breach caused by market movements must be corrected within six months "
                    "and the manager must inform the trustee (see the intimation date below).")
        st.error(msg)
    show_sections = compliance_sections_required(nbr, ruleset)

    if not show_sections:
        st.info("NBR is below the threshold. Credit Rating and Unitholder Approval sections are not required.")
        return False
    return True

def compliance_gaps(row, ruleset: str) -> list:
    """Credit rating / unitholder approval items that are required at this NBR but missing in the row."""
    cols = row.index
    cra1_rating = row.get(_find_col(cols, aliases=["Credit Rating CRA1"]))
    cra2_rating = row.get(_find_col(cols, aliases=["Credit Rating CRA2"]))
    ua_col = _find_col(cols, must_tokens=["unitholder", "approval"], exclude_tokens=["date", "meeting", "weblink"])
    unitholder_approval_val = row.get(ua_col)

    credit_taken_any = _is_taken(cra1_rating) or _is_taken(cra2_rating)
    aaa_ok = _is_aaa(cra1_rating) or _is_aaa(cra2_rating)
    unit_taken = _is_yes(unitholder_approval_val)

    nbr = row.get("NBR_ratio", 0.0) or 0.0
    return missing_compliance_items(nbr, ruleset, credit_taken_any, aaa_ok, unit_taken)


def _check_compliance_alerts(row, ruleset: str):
    missing = compliance_gaps(row, ruleset)
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


AREA = "Borrowings"


def summary_results(entity: str) -> list:
    """Scorecard verdicts for a REIT's latest reported quarter: the net-borrowings cap and, when the NBR is
    high enough to need them, credit rating and unitholder approval."""
    df = load_borrowings_url(DEFAULT_REIT_BORR_URL)
    rows = df[df[ENT_COL] == entity]
    fy = periods.latest_fy(rows[FY_COL].dropna().astype(str))
    if rows.empty or fy is None:
        return [CheckResult("Net borrowings", Status.NO_DATA, "No borrowings rows for this entity", AREA)]
    in_fy = rows[rows[FY_COL].astype(str) == fy]
    qtr = periods.latest_quarter(in_fy[QTR_COL].dropna().astype(str))
    row = in_fy[in_fy[QTR_COL].astype(str) == qtr].iloc[0]
    when = f"{fy} {qtr}"
    nbr = row.get("NBR_ratio", None)
    if not isinstance(nbr, (int, float)) or pd.isna(nbr):
        return [CheckResult("Net borrowings", Status.NO_DATA, f"{when}: NBR not available", AREA)]

    cap = nbr_cap("reit")
    over = nbr_over_cap(nbr, "reit")
    results = [CheckResult(
        f"Net borrowings within {cap*100:.0f}% cap", Status.FAIL if over else Status.PASS,
        f"{when}: NBR {nbr*100:.2f}%" + (f" exceeds the {cap*100:.0f}% cap (Reg. 20(2))" if over else ""), AREA, "borrowings.reit_cap")]
    if compliance_sections_required(nbr, "reit"):
        missing = compliance_gaps(row, "reit")
        results.append(CheckResult(
            "Credit rating and unitholder approval", Status.FAIL if missing else Status.PASS,
            f"{when}: NBR is {nbr*100:.2f}% (over {rules.REIT_NBR_TRIGGER*100:.0f}%), " + (f"missing: {', '.join(missing)}" if missing else "credit rating and unitholder approval are in place"),
            AREA, "borrowings.reit_trigger"))
    return results


def render():
    st.header("Borrowings")
    with st.sidebar:
        segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_borrow")
    
    # Auto-select URL (Hidden from UI)
    data_url = DEFAULT_INVIT_BORR_URL if segment == "InvIT" else DEFAULT_REIT_BORR_URL

    if not data_url.strip():
        st.warning("Please provide a data URL."); st.stop()

    try:
        df = load_borrowings_url(data_url.strip())
    except Exception as e:
        st.error(f"Could not read the URL. Make sure it’s publicly accessible.\n\nDetails: {e}"); st.stop()

    # Filters in Sidebar
    with st.sidebar:
        st.divider()
        entity = st.selectbox("Entity", sorted(df[ENT_COL].dropna().astype(str).unique()), key=f"entity_{segment}")
        fy = st.selectbox("Financial Year", sorted(df.loc[df[ENT_COL] == entity, FY_COL].dropna().astype(str).unique()), key=f"fy_{segment}")
        
        qtr_present = df.loc[(df[ENT_COL] == entity) & (df[FY_COL] == fy), QTR_COL].dropna().astype(str).unique().tolist()
        qtr = st.selectbox("Quarter", _quarter_sort(qtr_present), key=f"qtr_{segment}")

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
        how = str(row.get("NBR_how") or "")
        if how.startswith(("closest", "same entity", "assumed")):
            st.caption(f"The NBR in the sheet has no % sign; its unit was inferred ({how}).")
        computed = row.get("NBR_computed")
        if (isinstance(nbr, (int, float)) and not pd.isna(nbr) and isinstance(computed, (int, float))
                and not pd.isna(computed) and how != "computed from the components" and abs(nbr - computed) > 0.02):
            st.caption(f"Note: the NBR in the sheet ({nbr*100:.2f}%) differs from the NBR computed from its own components ({computed*100:.2f}%).")
    with colB:
        _render_card_breakup(row, df.attrs.get("__matched_cols__", {}))

    st.markdown("---")

    ruleset = "invit" if segment == "InvIT" else "reit"
    if _alerts_and_sections(row, ruleset):
        _check_compliance_alerts(row, ruleset)
        _render_credit_rating_ui(row)
        _render_unitholder_and_compliances_ui(row)