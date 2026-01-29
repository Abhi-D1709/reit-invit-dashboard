# tabs/related_party.py
import pandas as pd
import streamlit as st
from utils.common import (
    RPT_REIT_SHEET_URL,
    inject_global_css
)

@st.cache_data(ttl=600, show_spinner="Loading Related Party Data...")
def load_rpt_data():
    """
    Loads Sheet1 and Sheet2 from the RPT Google Sheet.
    Uses Excel export to fetch sheets by name.
    """
    # Convert edit URL to export URL for Excel
    url = RPT_REIT_SHEET_URL.replace("/edit?usp=sharing", "/export?format=xlsx")
    
    try:
        # Load all sheets
        all_sheets = pd.read_excel(url, sheet_name=None)
        
        cleaned = {}
        for name, df in all_sheets.items():
            # Standardize columns: strip whitespace
            df.columns = [str(c).strip() for c in df.columns]
            # Drop completely empty rows
            df.dropna(how='all', inplace=True)
            cleaned[name] = df
            
        return cleaned
    except Exception as e:
        st.error(f"Failed to load RPT Data: {e}")
        return {}

def render():
    st.header("Related Party Transactions")
    inject_global_css()

    # 1. Load Data
    sheets = load_rpt_data()
    if not sheets:
        return

    df_s1 = sheets.get("Sheet1", pd.DataFrame())
    df_s2 = sheets.get("Sheet2", pd.DataFrame())

    if df_s1.empty:
        st.warning("Sheet1 data is empty.")

    # 2. Sidebar Filters
    with st.sidebar:
        st.subheader("RPT Controls")
        
        # Extract unique Entities (Combine from both sheets for completeness)
        entities = set()
        if "Name of REIT" in df_s1.columns:
            entities.update(df_s1["Name of REIT"].dropna().astype(str).unique())
        if "Name of REIT" in df_s2.columns:
            entities.update(df_s2["Name of REIT"].dropna().astype(str).unique())
            
        selected_entity = st.selectbox("Select Entity", sorted(list(entities)), key="rpt_entity")

        # Extract unique Financial Years
        fys = set()
        if "Financial Year" in df_s1.columns:
            fys.update(df_s1["Financial Year"].dropna().astype(str).unique())
        if "Financial Year" in df_s2.columns:
            fys.update(df_s2["Financial Year"].dropna().astype(str).unique())
            
        selected_fy = st.selectbox("Select Financial Year", ["All"] + sorted(list(fys)), key="rpt_fy")

    if not selected_entity:
        st.info("Please select an Entity from the sidebar.")
        return

    # -------------------------------------------------------------------------
    # SECTION 1: Sheet 1 Analysis (Ceased Relations)
    # -------------------------------------------------------------------------
    st.subheader("1. Ceased Related Parties")
    
    # Filter Logic for Sheet 1
    mask_s1 = df_s1["Name of REIT"] == selected_entity
    if selected_fy != "All" and "Financial Year" in df_s1.columns:
        mask_s1 &= df_s1["Financial Year"].astype(str) == str(selected_fy)
    
    df1_filtered = df_s1[mask_s1].copy()

    ceased_col = "Relation Ceased with effect from"
    reason_col = "Reason for Related Party Cease/Indentification (For Related Parties identifed/ceased post listing)"
    
    if ceased_col in df1_filtered.columns:
        # Check if ceased column has data
        is_ceased = (
            df1_filtered[ceased_col].notna() & 
            (df1_filtered[ceased_col].astype(str).str.strip() != "") & 
            (df1_filtered[ceased_col].astype(str).str.strip() != "-")
        )
        
        ceased_rows = df1_filtered[is_ceased]
        
        if not ceased_rows.empty:
            cols_to_show = ["Name of Related Party", "Relation with the Related Party", ceased_col]
            if reason_col in ceased_rows.columns:
                cols_to_show.append(reason_col)
                
            st.dataframe(ceased_rows[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info(f"No related parties ceased in {selected_fy} for {selected_entity}.")
    else:
        st.info("Column 'Relation Ceased with effect from' not found in Sheet1.")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION 2: Sheet 2 Analysis (Unitholder Approvals)
    # -------------------------------------------------------------------------
    st.subheader("2. Unitholder Approvals")
    
    if not df_s2.empty and "Name of REIT" in df_s2.columns:
        # Filter Logic for Sheet 2
        mask_s2 = df_s2["Name of REIT"] == selected_entity
        if selected_fy != "All" and "Financial Year" in df_s2.columns:
            mask_s2 &= df_s2["Financial Year"].astype(str) == str(selected_fy)
            
        df2_filtered = df_s2[mask_s2].copy()
        
        # Check if any transactions exist (implies approval was taken)
        if not df2_filtered.empty:
            st.error("Check this transaction further")
            st.dataframe(df2_filtered, use_container_width=True, hide_index=True)
        else:
            st.info(f"No Unitholder Approvals found in {selected_fy} for {selected_entity}.")
    else:
        st.warning("Sheet2 data is empty or missing 'Name of REIT' column.")