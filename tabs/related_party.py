# tabs/related_party.py
import pandas as pd
import streamlit as st
import datetime
from utils.common import (
    RPT_REIT_SHEET_URL,
    inject_global_css,
    load_table_url
)

def get_fy_from_date(date_val):
    """
    Helper to convert a date object or string (DD/MM/YYYY) to Financial Year string (e.g., '2022-23').
    """
    if pd.isna(date_val) or str(date_val).strip() in ["-", "", "nan"]:
        return None
    
    dt_obj = None
    try:
        # Try parsing DD/MM/YYYY
        dt_obj = pd.to_datetime(date_val, dayfirst=True)
    except:
        return None
        
    if dt_obj is None or pd.isna(dt_obj):
        return None
    
    # Logic: If Month > 3 (April onwards), FY starts in current year.
    # Else (Jan-Mar), FY started in previous year.
    year = dt_obj.year
    if dt_obj.month > 3:
        start_year = year
        end_year = year + 1
    else:
        start_year = year - 1
        end_year = year
        
    # Format as 'YYYY-YY'
    return f"{start_year}-{str(end_year)[-2:]}"

@st.cache_data(ttl=600, show_spinner="Loading Related Party Data...")
def load_rpt_data():
    # Load via Excel export to get all sheets
    xlsx_url = RPT_REIT_SHEET_URL.replace("/edit?usp=sharing", "/export?format=xlsx")
    try:
        all_sheets = pd.read_excel(xlsx_url, sheet_name=None)
        cleaned = {}
        for k, v in all_sheets.items():
            # Clean column names
            v.columns = [str(c).strip() for c in v.columns]
            cleaned[k.strip()] = v
        return cleaned
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}

def render():
    st.header("Related Party Transactions")
    inject_global_css()
    
    data = load_rpt_data()
    if not data:
        return

    # --- Prepare Sheet 1 Data ---
    s1 = data.get("Sheet1", pd.DataFrame())
    
    # Pre-calculate FY for Sheet 1 based on 'Relation Ceased' column for filtering
    if not s1.empty:
        cease_col = "Relation Ceased with effect from"
        if cease_col in s1.columns:
            s1["_computed_fy"] = s1[cease_col].apply(get_fy_from_date)
    
    # --- Sidebar Filters ---
    with st.sidebar:
        st.subheader("Filters")
        
        # 1. Entity Filter
        entities = sorted(s1["Name of REIT"].dropna().unique()) if "Name of REIT" in s1.columns else []
        selected_entity = st.selectbox("Select Entity", entities)
        
        # 2. Financial Year Filter
        # Gather unique FYs from the computed column in Sheet1
        # (In later steps, we will merge this with FYs from other sheets)
        available_fys = sorted(s1["_computed_fy"].dropna().unique())
        selected_fy = st.selectbox("Select Financial Year", ["All"] + available_fys)

    if not selected_entity:
        st.info("Please select an Entity to proceed.")
        return

    # --- Part 1: Ceased Related Parties (Sheet 1) ---
    st.subheader("1. Ceased Related Parties")
    
    if not s1.empty:
        # Filter by Entity
        df_filtered = s1[s1["Name of REIT"] == selected_entity].copy()
        
        # Filter by Financial Year (if not 'All')
        # Logic: We check if the 'Relation Ceased' date falls within the selected FY
        if selected_fy != "All":
            df_filtered = df_filtered[df_filtered["_computed_fy"] == selected_fy]
        
        # Filter: Only show rows where 'Relation Ceased' has data
        cease_col = "Relation Ceased with effect from"
        reason_col = "Reason for Related Party Cease/Indentification (For Related Parties identifed/ceased post listing)"
        
        # Keep rows where cease_col is not null/empty
        mask = df_filtered[cease_col].notna() & (df_filtered[cease_col].astype(str).str.strip() != "-")
        display_df = df_filtered[mask].copy()
        
        if not display_df.empty:
            cols_to_show = [
                "Name of Related Party", 
                "Relation with the Related Party", 
                cease_col, 
                reason_col
            ]
            # Ensure columns exist before displaying
            final_cols = [c for c in cols_to_show if c in display_df.columns]
            
            st.dataframe(
                display_df[final_cols], 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info(f"No related parties ceased in {selected_fy} for {selected_entity}.")
    else:
        st.write("No data found in Sheet1.")