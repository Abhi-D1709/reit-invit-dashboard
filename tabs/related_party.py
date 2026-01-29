# tabs/related_party.py
import pandas as pd
import streamlit as st
from utils.common import (
    RPT_REIT_SHEET_URL,
    inject_global_css,
    load_table_url
)

def load_sheet1_data():
    """
    Loads Sheet1 from the RPT Google Sheet.
    """
    # Construct the CSV export URL for Sheet1 (gid=0 usually, or first sheet)
    # We use the base URL from common.py and ensure we fetch the first sheet if GID not specified,
    # or we can rely on pandas to fetch the first sheet by default if we use the export link.
    
    # Using the helper to ensure we get a CSV link
    # If RPT_REIT_SHEET_URL is a standard edit link, we convert it.
    url = RPT_REIT_SHEET_URL.replace("/edit?usp=sharing", "/export?format=csv&gid=0")
    
    try:
        df = pd.read_csv(url)
        # Standardize columns: strip whitespace
        df.columns = [str(c).strip() for c in df.columns]
        # Drop completely empty rows
        df.dropna(how='all', inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to load Sheet1: {e}")
        return pd.DataFrame()

def render():
    st.header("Related Party Transactions")
    inject_global_css()

    # 1. Load Data
    df_s1 = load_sheet1_data()
    
    if df_s1.empty:
        st.warning("Sheet1 data could not be loaded or is empty.")
        return

    # 2. Sidebar Filters
    with st.sidebar:
        st.subheader("RPT Controls")
        
        # Extract unique Entities
        if "Name of REIT" in df_s1.columns:
            entities = sorted(df_s1["Name of REIT"].dropna().astype(str).unique().tolist())
        else:
            entities = []
            
        selected_entity = st.selectbox("Select Entity", entities, key="rpt_entity")

        # Extract unique Financial Years
        # Note: We filter for non-null FYs to populate the dropdown
        if "Financial Year" in df_s1.columns:
            fys = sorted(df_s1["Financial Year"].dropna().astype(str).unique().tolist())
        else:
            fys = []
            
        selected_fy = st.selectbox("Select Financial Year", ["All"] + fys, key="rpt_fy")

    if not selected_entity:
        st.info("Please select an Entity from the sidebar.")
        return

    # 3. Filter Logic for Sheet1
    # "If for the selected entity, for the selected Financial Year..."
    
    # Filter by Entity
    mask = df_s1["Name of REIT"] == selected_entity
    
    # Filter by Financial Year (if not All)
    # We strictly enforce the FY check as requested
    if selected_fy != "All":
        if "Financial Year" in df_s1.columns:
            # Convert both to string for safe comparison
            mask &= df_s1["Financial Year"].astype(str) == str(selected_fy)
    
    filtered_df = df_s1[mask].copy()

    # 4. Display Logic
    # "If 'Relation Ceased with effect from' has any data..."
    
    st.subheader("1. Ceased Related Parties")
    
    ceased_col = "Relation Ceased with effect from"
    reason_col = "Reason for Related Party Cease/Indentification (For Related Parties identifed/ceased post listing)"
    
    if ceased_col in filtered_df.columns:
        # Check if ceased column has data (not null, not empty, not just a dash)
        # We assume "-" might be used for active parties based on your example
        is_ceased = (
            filtered_df[ceased_col].notna() & 
            (filtered_df[ceased_col].astype(str).str.strip() != "") & 
            (filtered_df[ceased_col].astype(str).str.strip() != "-")
        )
        
        ceased_rows = filtered_df[is_ceased]
        
        if not ceased_rows.empty:
            # Display specific columns
            cols_to_show = ["Name of Related Party", "Relation with the Related Party", ceased_col]
            if reason_col in ceased_rows.columns:
                cols_to_show.append(reason_col)
                
            st.dataframe(ceased_rows[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info(f"No related parties ceased in {selected_fy} for {selected_entity}.")
    else:
        st.error(f"Column '{ceased_col}' not found in Sheet1 data.")