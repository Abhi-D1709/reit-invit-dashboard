# tabs/investment.py
import pandas as pd
import streamlit as st
import re
from utils.common import (
    INVESTMENT_REIT_SHEET_URL,
    inject_global_css
)

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

def clean_currency(x):
    if pd.isna(x): return 0.0
    s = str(x).strip()
    if s in {"", "-", "NA", "N/A"}: return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return 0.0

def clean_percent(x):
    if pd.isna(x): return 0.0
    s = str(x).strip().replace("%", "")
    if s in {"", "-", "NA", "N/A"}: return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return 0.0

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
    if c_col and u_col:
        def check_80_rule(row):
            val_c = clean_currency(row[c_col])
            val_u = clean_currency(row[u_col])
            if val_u == 0: return "N/A"
            ratio = (val_c / val_u) * 100
            if 81 <= ratio <= 85: return f"🔴 {ratio:.2f}% (Warning: In 81-85% Bracket)"
            if ratio >= 80: return f"🟢 {ratio:.2f}% (Compliant)"
            return f"🔴 {ratio:.2f}% (Non-Compliant: < 80%)"

        filtered["Asset Ratio Check"] = filtered.apply(check_80_rule, axis=1)
        cols_1 = ["Name of REIT", "Financial Year", c_col, u_col, "Asset Ratio Check"]
        st.dataframe(filtered[cols_1].astype(str), use_container_width=True, hide_index=True)
        if filtered["Asset Ratio Check"].str.contains("🔴").any():
            st.error("Alert: Investment ratio issues found.")
        else:
            st.success("Asset Investment Ratios are healthy.")
    else:
        st.warning("Could not identify Columns C or U.")

    st.divider()

    # 2 & 3. SPV Checks
    st.subheader("2. SPV & Shareholder Agreement Checks")
    if y_col:
        spv_rows = filtered[filtered[y_col].astype(str).str.lower() == "yes"].copy()
        if not spv_rows.empty:
            st.warning("⚠️ 'Yes' found in SPV < 100% Equity column.")
            st.info("Action: Check Shareholder Agreement.") 
            if len(spv_hold_cols) >= 1:
                def check_holdings(row):
                    issues = []
                    for col in spv_hold_cols:
                        val = clean_percent(row[col])
                        if val > 50: issues.append(f"{col}: {val}% (> 50%)")
                    if issues: return "🔴 " + ", ".join(issues)
                    return "🟢 All <= 50%"

                spv_rows["Holding Check"] = spv_rows.apply(check_holdings, axis=1)
                show_spv_cols = ["Name of REIT", "Financial Year", y_col] + spv_hold_cols + ["Holding Check"]
                st.dataframe(spv_rows[show_spv_cols].astype(str), use_container_width=True, hide_index=True)
                if spv_rows["Holding Check"].str.contains("🔴").any():
                    st.error("Alert: Some SPV holdings exceed 50%.")
                else:
                    st.success("All SPV holdings are ≤ 50%.")
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
        has_data = (
            filtered[r_col].notna() & 
            (filtered[r_col].astype(str).str.strip() != "") & 
            (filtered[r_col].astype(str).str.strip() != "-")
        ).any()
        val_sum = filtered[r_col].apply(clean_currency).sum()
        
        if has_data and val_sum > 0:
            st.dataframe(filtered[["Name of REIT", "Financial Year", r_col]].astype(str), use_container_width=True, hide_index=True)
            st.warning("⚠️ Alert: Mutual Fund investments found. Check the credit risk value and class of mutual funds.")
        else:
            st.success("No Mutual Fund investments found.")
    else:
        st.warning("Could not identify Column R (Mutual Funds).")