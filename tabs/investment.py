# tabs/investment.py
import pandas as pd
import streamlit as st
import re
from utils.common import (
    INVESTMENT_REIT_SHEET_URL,
    inject_global_css,
    parse_number,
    resolve_percent_units,
    _MISSING_TEXT,
)
from utils import rules  # every threshold lives in utils/rules.py

NO_DATA = "⚪ N/A — insufficient data"


def asset_ratio_status(completed: float, total: float) -> str:
    """Status text for "investment in completed assets >= 80% of total REIT assets".

    A missing figure (blank, "-", "NA") is missing, not zero: reading it as 0 turned every row
    without a value into "Alert: < 80%". The 81-85% band is kept as configured on this page."""
    if pd.isna(completed) or pd.isna(total) or total == 0:
        return NO_DATA
    ratio = (completed / total) * 100
    minimum = rules.INVEST_COMPLETED_MIN_PCT
    low, high = rules.INVEST_ALERT_BAND
    if low <= ratio <= high:
        return f"🔴 {ratio:.2f}% (Alert: In {low:g}-{high:g}% Bracket)"
    if ratio >= minimum:
        return f"🟢 {ratio:.2f}% (No alert)"
    return f"🔴 {ratio:.2f}% (Alert: < {minimum:g}%)"

def spv_holding_status(holdings: dict) -> str:
    """Status text for the SPV shareholder check. `holdings` maps column name -> holding in percent points,
    only for the columns that have a figure. No figures at all is "no data", not a pass."""
    if not holdings:
        return "⚪ No holding figures provided"
    limit = rules.SPV_HOLDING_MAX_PCT
    issues = [f"{col}: {val:g}% (> {limit:g}%)" for col, val in holdings.items() if val > limit]
    if issues:
        return "🔴 " + ", ".join(issues)
    return f"🟢 All <= {limit:g}%"


@st.cache_data(ttl=600, show_spinner="Loading Investment Data...")
def load_investment_data():
    url = INVESTMENT_REIT_SHEET_URL.replace("/edit?usp=sharing", "/export?format=xlsx")
    try:
        df = pd.read_excel(url, sheet_name=0)
        df.columns = [str(c).strip() for c in df.columns]
        df.dropna(how='all', inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to load Investment Data: {e}")
        return pd.DataFrame()

def render():
    st.header("Investment Conditions")
    inject_global_css()

    df = load_investment_data()
    if df.empty:
        st.warning("No data found in Investment Sheet.")
        return

    with st.sidebar:
        st.subheader("Investment Controls")
        entities = sorted(df["Name of REIT"].dropna().astype(str).unique()) if "Name of REIT" in df.columns else []
        sel_entity = st.selectbox("Select Entity", entities, key="inv_ent")
        fys = sorted(df["Financial Year"].dropna().astype(str).unique()) if "Financial Year" in df.columns else []
        sel_fy = st.selectbox("Select Financial Year", ["All"] + fys, key="inv_fy")

    if not sel_entity:
        st.info("Please select an Entity.")
        return

    mask = df["Name of REIT"] == sel_entity
    if sel_fy != "All":
        mask &= df["Financial Year"].astype(str) == str(sel_fy)
    
    filtered = df[mask].copy()
    if filtered.empty:
        st.info("No records found for selection.")
        return

    cols = filtered.columns
    def get_col(keywords):
        for c in cols:
            if all(k.lower() in c.lower() for k in keywords):
                return c
        return None

    c_col = get_col(["completed", "rent generating", "investments"]) 
    u_col = get_col(["total value", "reit assets"]) 
    r_col = get_col(["mutual funds", "credit risk"]) 
    y_col = get_col(["spv", "less than 100", "equity"]) 
    spv_hold_cols = [c for c in cols if "% holding" in c.lower() and "sh" in c.lower()]

    # 1. Asset Ratio Check
    st.subheader("1. Investment in Completed Assets (≥ 80%)")
    band_lo, band_hi = rules.INVEST_ALERT_BAND
    st.caption(
        f"Rules: Target ≥ {rules.INVEST_COMPLETED_MIN_PCT:g}% (Green; REIT Regulations, Reg. 18(4)). "
        f"**Extra alert:** Red if the ratio is between {band_lo:g}% and {band_hi:g}% (a dashboard alert, not a regulatory limit)."
    )
    
    if c_col and u_col:
        filtered["Asset Ratio Check"] = [
            asset_ratio_status(parse_number(r[c_col]), parse_number(r[u_col])) for _, r in filtered.iterrows()
        ]
        cols_1 = ["Name of REIT", "Financial Year", c_col, u_col, "Asset Ratio Check"]
        
        st.dataframe(filtered[cols_1].astype(str), use_container_width=True, hide_index=True)
        
        # Check if any row triggered the specific 81-85% warning
        no_data = filtered["Asset Ratio Check"] == NO_DATA
        if filtered["Asset Ratio Check"].str.contains("Bracket").any():
            st.error(f"Alert: Some investments fall within the {band_lo:g}-{band_hi:g}% warning bracket.")
        elif filtered["Asset Ratio Check"].str.contains("🔴").any():
            st.error("Alert: Investment ratio below 80%.")
        if no_data.any():
            st.info(f"{int(no_data.sum())} row(s) have no completed-assets or total-assets figure, so the ratio could not be checked.")
        elif not filtered["Asset Ratio Check"].str.contains("🔴").any():
            st.success("Asset Investment Ratios are compliant (≥ 80% and outside warning bracket).")
    else:
        st.warning("Could not identify Columns C or U.")

    st.divider()

    # 2 & 3. SPV Checks
    st.subheader("2. SPV & Shareholder Agreement Checks")
    if y_col:
        # Check for Yes (case-insensitive)
        spv_rows = filtered[filtered[y_col].astype(str).str.lower() == "yes"].copy()
        
        if not spv_rows.empty:
            st.warning("⚠️ 'Yes' found in SPV < 100% Equity column.")
            st.info("Action: Check Shareholder Agreement.") 
            
            if len(spv_hold_cols) >= 1:
                # percent points per column (the sheet may hold "60%", 60 or 0.6); blank stays blank
                points = {col: resolve_percent_units(spv_rows[col])[0] * 100 for col in spv_hold_cols}

                def check_holdings(row):
                    known = {col: points[col].loc[row.name] for col in spv_hold_cols if not pd.isna(points[col].loc[row.name])}
                    return spv_holding_status(known)

                spv_rows["Holding Check"] = spv_rows.apply(check_holdings, axis=1)

                show_spv_cols = ["Name of REIT", "Financial Year", y_col] + spv_hold_cols + ["Holding Check"]
                st.dataframe(spv_rows[show_spv_cols].astype(str), use_container_width=True, hide_index=True)

                if spv_rows["Holding Check"].str.contains("🔴").any():
                    st.error("Alert: Some SPV holdings exceed 50%.")
                if spv_rows["Holding Check"].str.contains("⚪").any():
                    st.info("Some rows give no SPV holding figures, so they could not be checked.")
                elif not spv_rows["Holding Check"].str.contains("🔴").any():
                    st.success(f"All SPV holdings are ≤ {rules.SPV_HOLDING_MAX_PCT:g}%.")
            else:
                st.warning("Could not find SPV Holding columns.")
        else:
            st.success("No SPVs with < 100% Equity found (Column Y is No).")
    else:
        st.warning("Could not identify Column Y.")

    st.divider()

    # 4. Mutual Funds
    st.subheader("3. Mutual Funds Credit Risk")
    if r_col:
        values = filtered[r_col].map(parse_number)
        # something is written in the cell (not blank / "-" / "NA") ...
        written = filtered[r_col].map(lambda x: not (pd.isna(x) or str(x).strip().lower() in _MISSING_TEXT))
        # ... but it isn't a readable number: still counts as "found", it must not pass as "no investment"
        unreadable = written & values.isna()

        if (values > 0).any() or unreadable.any():
            st.dataframe(filtered[["Name of REIT", "Financial Year", r_col]].astype(str), use_container_width=True, hide_index=True)
            st.warning("⚠️ Alert: Mutual Fund investments found. Check the credit risk value and class of mutual funds.")
            if unreadable.any():
                st.caption(f"{int(unreadable.sum())} value(s) could not be read as a number; please check them in the sheet.")
        else:
            st.success("No Mutual Fund investments found.")
    else:
        st.warning("Could not identify Column R (Mutual Funds).")