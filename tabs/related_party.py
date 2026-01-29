# tabs/related_party.py
import pandas as pd
import streamlit as st
from utils.common import (
    RPT_REIT_SHEET_URL,
    DEFAULT_REIT_BORR_URL,
    ENT_COL, FY_COL, QTR_COL,
    inject_global_css,
    load_table_url,
    _standardize_selector_columns
)

def clean_currency(x):
    """
    Helper to convert string currency (e.g., '3,16,124' or '25.325') to float.
    Returns 0.0 if conversion fails.
    """
    if pd.isna(x): return 0.0
    s = str(x).strip()
    if s in {"", "-", "NA", "N/A"}: return 0.0
    # Remove commas
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0

@st.cache_data(ttl=600, show_spinner="Loading Related Party Data...")
def load_rpt_data():
    """
    Loads Sheet1, Sheet2, Sheet3, Sheet4 from the RPT Google Sheet
    AND the Borrowings data for Asset Value lookups.
    """
    # 1. Load RPT Sheets via Excel export
    url = RPT_REIT_SHEET_URL.replace("/edit?usp=sharing", "/export?format=xlsx")
    
    rpt_sheets = {}
    borrowings_df = pd.DataFrame()
    
    try:
        all_sheets = pd.read_excel(url, sheet_name=None)
        for name, df in all_sheets.items():
            df.columns = [str(c).strip() for c in df.columns]
            df.dropna(how='all', inplace=True)
            rpt_sheets[name] = df
    except Exception as e:
        st.error(f"Failed to load RPT Data: {e}")

    # 2. Load Borrowings Data (for Asset Value)
    try:
        df_b = load_table_url(DEFAULT_REIT_BORR_URL)
        # Standardize entity/FY/Quarter columns using common util
        borrowings_df = _standardize_selector_columns(df_b)
    except Exception as e:
        st.error(f"Failed to load Borrowings Data: {e}")

    return rpt_sheets, borrowings_df

def render():
    st.header("Related Party Transactions")
    inject_global_css()

    # 1. Load Data
    sheets, borrowings_df = load_rpt_data()
    if not sheets:
        return

    df_s1 = sheets.get("Sheet1", pd.DataFrame())
    df_s2 = sheets.get("Sheet2", pd.DataFrame())
    df_s3 = sheets.get("Sheet3", pd.DataFrame())
    df_s4 = sheets.get("Sheet4", pd.DataFrame())

    if df_s1.empty:
        st.warning("Sheet1 data is empty.")

    # 2. Sidebar Filters
    with st.sidebar:
        st.subheader("RPT Controls")
        
        # Extract unique Entities (Combine from all relevant sheets)
        entities = set()
        for df in [df_s1, df_s2, df_s3, df_s4]:
            if "Name of REIT" in df.columns:
                entities.update(df["Name of REIT"].dropna().astype(str).unique())
            
        selected_entity = st.selectbox("Select Entity", sorted(list(entities)), key="rpt_entity")

        # Extract unique Financial Years (Combine from all relevant sheets)
        fys = set()
        for df in [df_s1, df_s2, df_s3]:
            if "Financial Year" in df.columns:
                fys.update(df["Financial Year"].dropna().astype(str).unique())
            
        selected_fy = st.selectbox("Select Financial Year", ["All"] + sorted(list(fys)), key="rpt_fy")

    if not selected_entity:
        st.info("Please select an Entity from the sidebar.")
        return

    # -------------------------------------------------------------------------
    # SECTION 1: Sheet 1 Analysis (Ceased Relations)
    # -------------------------------------------------------------------------
    st.subheader("1. Ceased Related Parties")
    
    # Filter Logic for Sheet 1
    if not df_s1.empty and "Name of REIT" in df_s1.columns:
        mask_s1 = df_s1["Name of REIT"] == selected_entity
        if selected_fy != "All" and "Financial Year" in df_s1.columns:
            mask_s1 &= df_s1["Financial Year"].astype(str) == str(selected_fy)
        
        df1_filtered = df_s1[mask_s1].copy()

        ceased_col = "Relation Ceased with effect from"
        reason_col = "Reason for Related Party Cease/Indentification (For Related Parties identifed/ceased post listing)"
        
        if ceased_col in df1_filtered.columns:
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
    else:
        st.info("No data available in Sheet1.")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION 2: Sheet 2 Analysis (Unitholder Approvals)
    # -------------------------------------------------------------------------
    st.subheader("2. Unitholder Approvals")
    
    if not df_s2.empty and "Name of REIT" in df_s2.columns:
        mask_s2 = df_s2["Name of REIT"] == selected_entity
        if selected_fy != "All" and "Financial Year" in df_s2.columns:
            mask_s2 &= df_s2["Financial Year"].astype(str) == str(selected_fy)
            
        df2_filtered = df_s2[mask_s2].copy()
        
        if not df2_filtered.empty:
            st.error("Check this transaction further")
            st.dataframe(df2_filtered, use_container_width=True, hide_index=True)
        else:
            st.info(f"No Unitholder Approvals found in {selected_fy} for {selected_entity}.")
    else:
        st.warning("Sheet2 data is empty.")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION 3: Sheet 3 Analysis (RPT Intensity Metric)
    # -------------------------------------------------------------------------
    st.subheader("3. RPT Intensity Metric")
    st.caption("Total Value of RPT Transactions vs. Value of REIT Assets (End of FY)")

    if not df_s3.empty and "Name of REIT" in df_s3.columns and not borrowings_df.empty:
        # 1. Filter Transactions (Sheet 3)
        mask_s3 = df_s3["Name of REIT"] == selected_entity
        if selected_fy != "All":
            mask_s3 &= df_s3["Financial Year"].astype(str) == str(selected_fy)
        
        df3_filtered = df_s3[mask_s3].copy()
        
        # 2. Display Transactions Table
        if not df3_filtered.empty:
            st.dataframe(df3_filtered, use_container_width=True, hide_index=True)
        
        # 3. Calculate Metric
        if selected_fy == "All":
            st.info("Please select a specific Financial Year to calculate the RPT Intensity Metric.")
        else:
            # A. Total RPT Value
            rpt_col = "Amount of Transaction"
            if rpt_col in df3_filtered.columns:
                total_rpt_value = df3_filtered[rpt_col].apply(clean_currency).sum()
                
                # B. Value of REIT Assets (from Borrowings)
                # Filter Borrowings by Entity + FY + Quarter containing "Mar"
                mask_borr = (
                    (borrowings_df[ENT_COL] == selected_entity) & 
                    (borrowings_df[FY_COL].astype(str) == str(selected_fy)) &
                    (borrowings_df[QTR_COL].astype(str).str.contains("Mar", case=False, na=False))
                )
                
                assets_row = borrowings_df[mask_borr]
                
                if not assets_row.empty:
                    asset_val_raw = assets_row.iloc[0]["Value of REIT Assets"]
                    asset_val = clean_currency(asset_val_raw)
                    
                    if asset_val > 0:
                        percentage = (total_rpt_value / asset_val) * 100
                        
                        # Display Metrics
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total RPT Value", f"{total_rpt_value:,.2f}")
                        c2.metric("REIT Assets (Mar)", f"{asset_val:,.2f}")
                        c3.metric("RPT Intensity", f"{percentage:.2f}%")
                    else:
                        st.warning(f"Value of REIT Assets is 0 for {selected_entity} in FY {selected_fy}.")
                else:
                    st.warning(f"Could not find Asset Value for {selected_entity} (FY {selected_fy}, Quarter 'Mar') in Borrowings data.")
            else:
                st.warning(f"Column '{rpt_col}' not found in Sheet3.")
    else:
        if df_s3.empty: st.info("Sheet3 data is empty.")
        if borrowings_df.empty: st.error("Borrowings data could not be loaded.")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION 4: Sheet 4 Analysis (Lease Transactions)
    # -------------------------------------------------------------------------
    st.subheader("4. Lease Transactions")
    
    if not df_s4.empty and "Name of REIT" in df_s4.columns:
        # Filter Logic for Sheet 4
        mask_s4 = df_s4["Name of REIT"] == selected_entity
        
        # Note: Sheet 4 typically does not have a Financial Year column in the sample,
        # but if it does, we respect the filter.
        if selected_fy != "All" and "Financial Year" in df_s4.columns:
            mask_s4 &= df_s4["Financial Year"].astype(str) == str(selected_fy)
            
        df4_filtered = df_s4[mask_s4].copy()
        
        # Check Column H: "If Yes, Date of Untiholder Approval"
        # "If yes [entry exists], then Red alert, else green."
        target_col = "If Yes, Date of Untiholder Approval"
        
        if target_col in df4_filtered.columns:
            # Display the data first
            st.dataframe(df4_filtered, use_container_width=True, hide_index=True)
            
            # Check for non-empty entries in Column H
            has_entry = (
                df4_filtered[target_col].notna() & 
                (df4_filtered[target_col].astype(str).str.strip() != "") & 
                (df4_filtered[target_col].astype(str).str.strip() != "-")
            ).any()
            
            if has_entry:
                st.error("Alert: Unitholder Approval entry found (Column H).")
            else:
                st.success("No Unitholder Approval entries found (Column H).")
        else:
            st.warning(f"Column '{target_col}' not found in Sheet4.")
            # Fallback: Just show the table if column missing
            st.dataframe(df4_filtered, use_container_width=True, hide_index=True)
    else:
        st.info("No data available in Sheet4.")