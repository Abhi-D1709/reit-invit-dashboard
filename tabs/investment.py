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
    # Convert to Excel export to safely get the sheet
    url = INVESTMENT_REIT_SHEET_URL.replace("/edit?usp=sharing", "/export?format=xlsx")
    try:
        df = pd.read_excel(url, sheet_name=0)
        # Clean header names
        df.columns = [str(c).strip() for c in df.columns]
        # Drop empty rows
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
    """
    Parses "50%" -> 50.0, "50" -> 50.0. 
    Returns 0.0 if empty/dash.
    """
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

    # --- Filters ---
    with st.sidebar:
        st.subheader("Investment Controls")
        
        entities = sorted(df["Name of REIT"].dropna().astype(str).unique()) if "Name of REIT" in df.columns else []
        sel_entity = st.selectbox("Select Entity", entities, key="inv_ent")
        
        fys = sorted(df["Financial Year"].dropna().astype(str).unique()) if "Financial Year" in df.columns else []
        sel_fy = st.selectbox("Select Financial Year", ["All"] + fys, key="inv_fy")

    if not sel_entity:
        st.info("Please select an Entity.")
        return

    # Filter Data
    mask = df["Name of REIT"] == sel_entity
    if sel_fy != "All":
        mask &= df["Financial Year"].astype(str) == str(sel_fy)
    
    filtered = df[mask].copy()
    
    if filtered.empty:
        st.info("No records found for selection.")
        return

    # Column Mapping (indices based on provided snippet order)
    # Using name-based lookups where possible to be safer, but falling back to keywords
    cols = filtered.columns
    
    # Identify key columns using flexible matching
    # C: Completed Rent Generating
    # U: Total Value of REIT Assets
    # R: Mutual Funds (Risk >= 12)
    # Y: SPV < 100% Equity?
    
    # Helper to find column by keyword
    def get_col(keywords):
        for c in cols:
            if all(k.lower() in c.lower() for k in keywords):
                return c
        return None

    c_col = get_col(["completed", "rent generating", "investments"]) # Column C
    u_col = get_col(["total value", "reit assets"]) # Column U
    r_col = get_col(["mutual funds", "credit risk"]) # Column R
    y_col = get_col(["spv", "less than 100", "equity"]) # Column Y
    
    # SPV Holdings (AB, AE, AH) - Harder to match by name due to duplicates in source
    # We will try to find indices relative to 'y_col' if names fail or are duplicate
    # " (% Holding of other SH in SPV)" appears multiple times
    spv_hold_cols = [c for c in cols if "% holding" in c.lower() and "sh" in c.lower()]

    # -------------------------------------------------------------------------
    # Logic 1: Completed Assets Ratio (C >= 80% of U)
    # -------------------------------------------------------------------------
    st.subheader("1. Investment in Completed Assets (≥ 80%)")
    
    if c_col and u_col:
        def check_80_rule(row):
            val_c = clean_currency(row[c_col])
            val_u = clean_currency(row[u_col])
            
            if val_u == 0: return "N/A"
            
            ratio = (val_c / val_u) * 100
            
            # Logic 5: Bracket Check (81-85%)
            if 81 <= ratio <= 85:
                return f"🔴 {ratio:.2f}% (Warning: In 81-85% Bracket)"
            
            # Logic 1: >= 80%
            if ratio >= 80:
                return f"🟢 {ratio:.2f}% (Compliant)"
            else:
                return f"🔴 {ratio:.2f}% (Non-Compliant: < 80%)"

        filtered["Asset Ratio Check"] = filtered.apply(check_80_rule, axis=1)
        
        # Display
        cols_1 = ["Name of REIT", "Financial Year", c_col, u_col, "Asset Ratio Check"]
        st.dataframe(filtered[cols_1], use_container_width=True, hide_index=True)
        
        # Alerts
        if filtered["Asset Ratio Check"].str.contains("🔴").any():
            st.error("Alert: Investment ratio issues found (Either <80% or in 81-85% warning bracket).")
        else:
            st.success("Asset Investment Ratios are healthy.")
    else:
        st.warning("Could not identify Columns C or U.")

    st.divider()

    # -------------------------------------------------------------------------
    # Logic 2 & 3: SPV Holding Checks (If Y = Yes)
    # -------------------------------------------------------------------------
    st.subheader("2. SPV & Shareholder Agreement Checks")
    
    if y_col:
        # Filter for rows where Y = Yes
        spv_rows = filtered[filtered[y_col].astype(str).str.lower() == "yes"].copy()
        
        if not spv_rows.empty:
            st.warning("⚠️ 'Yes' found in SPV < 100% Equity column.")
            st.info("Action: Check Shareholder Agreement.") # Logic 3
            
            # Logic 2: Check Hold % <= 50%
            # We assume spv_hold_cols[0], [1], [2] correspond to AB, AE, AH
            if len(spv_hold_cols) >= 1:
                def check_holdings(row):
                    issues = []
                    for col in spv_hold_cols:
                        val = clean_percent(row[col])
                        # If val is 0, likely empty, skip check? Or assume compliant?
                        # Assuming empty means no SPV 2/3 exists.
                        if val > 50:
                            issues.append(f"{col}: {val}% (> 50%)")
                    
                    if issues:
                        return "🔴 " + ", ".join(issues)
                    return "🟢 All <= 50%"

                spv_rows["Holding Check"] = spv_rows.apply(check_holdings, axis=1)
                
                # Show relevant columns
                show_spv_cols = ["Name of REIT", "Financial Year", y_col] + spv_hold_cols + ["Holding Check"]
                st.dataframe(spv_rows[show_spv_cols], use_container_width=True, hide_index=True)
                
                if spv_rows["Holding Check"].str.contains("🔴").any():
                    st.error("Alert: Some SPV holdings exceed 50%.")
                else:
                    st.success("All SPV holdings are ≤ 50%.")
            else:
                st.warning("Could not find SPV Holding columns (AB, AE, AH) to check percentages.")
        else:
            st.success("No SPVs with < 100% Equity found (Column Y is No).")
    else:
        st.warning("Could not identify Column Y (SPV < 100% Equity).")

    st.divider()

    # -------------------------------------------------------------------------
    # Logic 4: Mutual Funds Check (Column R)
    # -------------------------------------------------------------------------
    st.subheader("3. Mutual Funds Credit Risk")
    
    if r_col:
        # Check if R has any data (not null/dash)
        # We check the raw value first
        has_data = (
            filtered[r_col].notna() & 
            (filtered[r_col].astype(str).str.strip() != "") & 
            (filtered[r_col].astype(str).str.strip() != "-")
        ).any()
        
        # We also check if numerical value > 0 just in case
        val_sum = filtered[r_col].apply(clean_currency).sum()
        
        if has_data and val_sum > 0:
            st.dataframe(filtered[["Name of REIT", "Financial Year", r_col]], use_container_width=True, hide_index=True)
            st.warning("⚠️ Alert: Mutual Fund investments found. Check the credit risk value and class of mutual funds.")
        else:
            st.success("No Mutual Fund investments found (Column R is empty/zero).")
    else:
        st.warning("Could not identify Column R (Mutual Funds).")