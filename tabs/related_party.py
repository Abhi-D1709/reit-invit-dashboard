# tabs/related_party.py
import pandas as pd
import streamlit as st
from utils.common import (
    RPT_REIT_SHEET_URL,
    inject_global_css
)

@st.cache_data(ttl=600, show_spinner="Loading Related Party Transactions...")
def load_all_rpt_sheets():
    """
    Loads the entire RPT workbook (all sheets) at once via Excel export.
    This is more robust than guessing GIDs for Sheet1, Sheet2, etc.
    """
    # Convert edit URL to export URL
    xlsx_url = RPT_REIT_SHEET_URL.replace("/edit?usp=sharing", "/export?format=xlsx")
    
    try:
        # Load all sheets as a dictionary of DataFrames
        all_sheets = pd.read_excel(xlsx_url, sheet_name=None)
        
        # Clean up column names and empty rows for each sheet
        cleaned = {}
        for sheet_name, df in all_sheets.items():
            # Standardize headers: strip whitespace
            df.columns = [str(c).strip() for c in df.columns]
            # Drop rows where all elements are NaN
            df.dropna(how='all', inplace=True)
            cleaned[sheet_name] = df
            
        return cleaned
    except Exception as e:
        st.error(f"Failed to load RPT Data: {e}")
        return {}

def render():
    st.header("Related Party Transactions")
    inject_global_css()

    # 1. Load All Data
    sheets = load_all_rpt_sheets()
    if not sheets:
        return

    # Extract specific sheets (defaulting to empty DF if missing)
    df_s1 = sheets.get("Sheet1", pd.DataFrame())
    df_s2 = sheets.get("Sheet2", pd.DataFrame())

    # 2. Sidebar Filters
    with st.sidebar:
        st.subheader("RPT Controls")
        
        # Get Entities from Sheet1 or Sheet2
        entities = set()
        if "Name of REIT" in df_s1.columns:
            entities.update(df_s1["Name of REIT"].dropna().astype(str).unique())
        if "Name of REIT" in df_s2.columns:
            entities.update(df_s2["Name of REIT"].dropna().astype(str).unique())
            
        selected_entity = st.selectbox("Select Entity", sorted(list(entities)), key="rpt_entity")

        # Get Financial Years from Sheet2 (Sheet1 usually doesn't have FY for all rows, but Sheet2 does)
        fys = set()
        if "Financial Year" in df_s2.columns:
            fys.update(df_s2["Financial Year"].dropna().astype(str).unique())
            
        selected_fy = st.selectbox("Select Financial Year", ["All"] + sorted(list(fys)), key="rpt_fy")

    if not selected_entity:
        st.info("Please select an Entity from the sidebar.")
        return

    # =========================================================================
    # PART 1: Ceased Related Parties (Sheet 1)
    # =========================================================================
    st.subheader("1. Ceased Related Parties")
    
    if not df_s1.empty and "Name of REIT" in df_s1.columns:
        # Filter by Entity
        mask_s1 = df_s1["Name of REIT"] == selected_entity
        
        # Note: Sheet1 in your example sometimes has a 'Financial Year' column for specific events,
        # but often lists static party details. If the column exists, we can filter, 
        # but usually Ceased Date is the key. 
        # We will filter by FY only if the column exists and user didn't select "All".
        if selected_fy != "All" and "Financial Year" in df_s1.columns:
             mask_s1 &= df_s1["Financial Year"].astype(str) == str(selected_fy)
             
        df1_filtered = df_s1[mask_s1].copy()
        
        ceased_col = "Relation Ceased with effect from"
        reason_col = "Reason for Related Party Cease/Indentification (For Related Parties identifed/ceased post listing)"
        
        if ceased_col in df1_filtered.columns:
            # Check for non-empty cease dates
            is_ceased = (
                df1_filtered[ceased_col].notna() & 
                (df1_filtered[ceased_col].astype(str).str.strip() != "") & 
                (df1_filtered[ceased_col].astype(str).str.strip() != "-")
            )
            ceased_rows = df1_filtered[is_ceased]
            
            if not ceased_rows.empty:
                cols_show = ["Name of Related Party", "Relation with the Related Party", ceased_col]
                if reason_col in ceased_rows.columns:
                    cols_show.append(reason_col)
                st.dataframe(ceased_rows[cols_show], use_container_width=True, hide_index=True)
            else:
                st.info(f"No ceased related parties found for {selected_entity}.")
        else:
            st.info("No ceased date column found in Sheet1.")
    else:
        st.info("No data available in Sheet1.")

    st.divider()

    # =========================================================================
    # PART 2: Unitholder Approvals (Sheet 2)
    # =========================================================================
    st.subheader("2. Unitholder Approvals")

    if not df_s2.empty and "Name of REIT" in df_s2.columns:
        # Filter by Entity
        mask_s2 = df_s2["Name of REIT"] == selected_entity
        
        # Filter by FY
        if selected_fy != "All" and "Financial Year" in df_s2.columns:
            mask_s2 &= df_s2["Financial Year"].astype(str) == str(selected_fy)
            
        df2_filtered = df_s2[mask_s2].copy()
        
        # Requirement: "check whether any unitholder approval has been taken... If yes, red alert"
        if not df2_filtered.empty:
            st.error("Check this transaction further")
            st.dataframe(df2_filtered, use_container_width=True, hide_index=True)
        else:
            st.success(f"No Unitholder Approvals found for {selected_entity} ({selected_fy}).")
    else:
        st.info("No data available in Sheet2.")