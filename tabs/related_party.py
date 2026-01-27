# tabs/related_party.py
import pandas as pd
import streamlit as st
import re
from utils.common import (
    DEFAULT_REIT_BORR_URL,
    RPT_REIT_SHEET_URL,
    load_table_url,
    inject_global_css,
    _standardize_selector_columns,
    _find_col,
    _num_series
)

@st.cache_data(ttl=600, show_spinner="Loading Related Party Data...")
def load_rpt_data():
    """
    Loads all sheets (Sheet1 to Sheet6) from the RPT Google Sheet
    and the Borrowings sheet for Asset Value lookups.
    """
    # 1. Load RPT Sheets via Excel export to get all sheets at once
    # This avoids guessing GIDs for 6 different sheets
    # UPDATED: Uses the new REIT specific variable
    xlsx_url = RPT_REIT_SHEET_URL.replace("/edit?usp=sharing", "/export?format=xlsx")
    
    try:
        # sheet_name=None loads all sheets as a dict of DataFrames
        all_sheets = pd.read_excel(xlsx_url, sheet_name=None)
        
        # Normalize sheet names (Sheet1, Sheet2...) and clean columns
        cleaned_sheets = {}
        for name, df in all_sheets.items():
            # Standardize column names: strip spaces
            df.columns = [str(c).strip() for c in df.columns]
            # Drop empty rows
            df.dropna(how='all', inplace=True)
            cleaned_sheets[name.strip()] = df
            
    except Exception as e:
        st.error(f"Failed to load RPT Google Sheet: {e}")
        return None, None

    # 2. Load Borrowings Sheet (for 'Value of REIT Assets')
    try:
        borrowings_df = load_table_url(DEFAULT_REIT_BORR_URL)
        borrowings_df = _standardize_selector_columns(borrowings_df)
    except Exception:
        borrowings_df = pd.DataFrame()

    return cleaned_sheets, borrowings_df

def clean_currency(x):
    """Helper to convert string currency to float."""
    if pd.isna(x): return 0.0
    s = str(x).replace(",", "").strip()
    try:
        return float(s)
    except:
        return 0.0

def render():
    st.header("Related Party Transactions")
    inject_global_css()

    sheets, borrowings = load_rpt_data()
    if not sheets:
        return

    # Map generic sheet names to logical variables
    s1 = sheets.get("Sheet1", pd.DataFrame())
    s2 = sheets.get("Sheet2", pd.DataFrame())
    s3 = sheets.get("Sheet3", pd.DataFrame())
    s4 = sheets.get("Sheet4", pd.DataFrame())
    s5 = sheets.get("Sheet5", pd.DataFrame())
    s6 = sheets.get("Sheet6", pd.DataFrame())

    # --- Sidebar Filters ---
    with st.sidebar:
        st.subheader("RPT Controls")
        
        # Get list of Entities from Sheet1 (or union of all)
        entities = sorted(s1["Name of REIT"].dropna().unique().tolist()) if "Name of REIT" in s1.columns else []
        if not entities and "Name of REIT" in s2.columns:
            entities = sorted(s2["Name of REIT"].dropna().unique().tolist())
            
        selected_entity = st.selectbox("Select Entity", entities)
        
        # Get list of FYs from Sheet2 or Sheet3
        fys = []
        if "Financial Year" in s2.columns: fys.extend(s2["Financial Year"].dropna().unique().tolist())
        if "Financial Year" in s3.columns: fys.extend(s3["Financial Year"].dropna().unique().tolist())
        fys = sorted(list(set(fys)))
        
        selected_fy = st.selectbox("Select Financial Year", ["All"] + fys)

    if not selected_entity:
        st.info("Please select an Entity.")
        return

    # ================= SECTION 1: Party Status (Sheet 1) =================
    st.subheader("1. Ceased Related Parties")
    if not s1.empty and "Name of REIT" in s1.columns:
        # Filter
        df1 = s1[s1["Name of REIT"] == selected_entity].copy()
        
        # Check for ceased relations
        ceased_col = "Relation Ceased with effect from"
        reason_col = "Reason for Related Party Cease/Indentification (For Related Parties identifed/ceased post listing)"
        
        if ceased_col in df1.columns:
            # Logic: Show if 'ceased_col' has data (not empty/dash)
            mask = df1[ceased_col].astype(str).str.strip().replace({"-": "", "nan": ""}) != ""
            ceased_df = df1[mask].copy()
            
            if not ceased_df.empty:
                cols_show = ["Name of Related Party", "Relation with the Related Party", ceased_col, reason_col]
                # Filter columns that actually exist
                cols_show = [c for c in cols_show if c in ceased_df.columns]
                st.dataframe(ceased_df[cols_show], use_container_width=True, hide_index=True)
            else:
                st.info("No ceased related parties found for this entity.")
    else:
        st.write("No data in Sheet1.")
        
    st.divider()

    # ================= SECTION 2 & 3: Approvals & Transaction Value (Sheet 2 & 3) =================
    st.subheader("2. Unitholder Approvals & Transaction Values")
    
    # Logic: Check Sheet 2 for Approvals in Selected FY
    if not s2.empty and not s3.empty:
        # Filter Sheet 2
        df2 = s2[s2["Name of REIT"] == selected_entity].copy()
        if selected_fy != "All":
            df2 = df2[df2["Financial Year"] == selected_fy]
        
        # Filter Sheet 3 (Transactions)
        df3 = s3[s3["Name of REIT"] == selected_entity].copy()
        if selected_fy != "All":
            df3 = df3[df3["Financial Year"] == selected_fy]

        # 2a. Display Approvals
        if not df2.empty:
            st.markdown(f"**Unitholder Approvals ({selected_fy if selected_fy != 'All' else 'All Years'})**")
            st.dataframe(df2, use_container_width=True, hide_index=True)
            
            # If approvals exist, show linked transactions from Sheet 3
            if not df3.empty:
                st.markdown("👉 **Related Transactions (from Sheet 3):**")
                st.dataframe(df3, use_container_width=True, hide_index=True)
        else:
            st.info(f"No Unitholder Approvals found for {selected_entity} ({selected_fy}).")

        # 2b. Calculate RPT % of Asset Value
        # Req: Total Value of all RPT (Sheet3) / Value of REIT Assets (Borrowings Sheet - Mar Quarter)
        if selected_fy != "All" and not df3.empty:
            # Sum RPT
            rpt_col = "Amount of Transaction"
            if rpt_col in df3.columns:
                total_rpt = df3[rpt_col].apply(clean_currency).sum()
                
                # Fetch Asset Value
                # Filter borrowings for Entity + FY + Quarter=Mar
                if not borrowings.empty:
                    borr_mask = (
                        (borrowings["Entity"] == selected_entity) & 
                        (borrowings["Financial Year"] == selected_fy) & 
                        (borrowings["Quarter Ended"].astype(str).str.contains("Mar", case=False, na=False))
                    )
                    asset_row = borrowings[borr_mask]
                    
                    if not asset_row.empty:
                        asset_val = clean_currency(asset_row.iloc[0]["Value of REIT Assets"])
                        
                        if asset_val > 0:
                            ratio = (total_rpt / asset_val) * 100
                            
                            st.markdown("#### RPT Intensity Metric")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Total RPT Value (FY)", f"₹ {total_rpt:,.2f} Cr")
                            c2.metric("REIT Asset Value (March)", f"₹ {asset_val:,.2f} Cr")
                            c3.metric("RPT as % of Assets", f"{ratio:.2f}%")
                        else:
                            st.warning("REIT Asset Value is 0 or missing in Borrowings sheet.")
                    else:
                        st.warning(f"Could not find 'Value of REIT Assets' for {selected_entity} / {selected_fy} / March in Borrowings Data.")
    
    st.divider()

    # ================= SECTION 4: Leases (Sheet 4) =================
    st.subheader("3. Lease Overlaps")
    st.caption("Checks if parties involved in Leases (Sheet 4) appear in Acquisitions (Sheet 5).")
    
    if not s4.empty and not s5.empty:
        df4 = s4[s4["Name of REIT"] == selected_entity].copy()
        
        # Check overlaps
        if "Name of Related Party" in df4.columns and "Name of Related Party" in s5.columns:
            # Get Sheet 5 parties for this entity
            s5_parties = set(s5[s5["Name of REIT"] == selected_entity]["Name of Related Party"].str.strip().unique())
            
            def check_overlap(row):
                party = str(row["Name of Related Party"]).strip()
                if party in s5_parties:
                    return "🔴 Common Party Found"
                return "🟢 No Overlap"

            if not df4.empty:
                df4["Overlap Status"] = df4.apply(check_overlap, axis=1)
                st.dataframe(df4, use_container_width=True, hide_index=True)
                
                if df4["Overlap Status"].str.contains("🔴").any():
                    st.error("Alert: Some parties in Leases are also present in Acquisition/Disposal transactions.")
            else:
                st.info("No Lease data found.")
    
    st.divider()

    # ================= SECTION 5: Acquisition Valuation (Sheet 5) =================
    st.subheader("4. Acquisition Valuation Check")
    st.caption("Condition: Transaction Value <= 110% of Average(Valuation 1, Valuation 2)")
    
    if not s5.empty:
        df5 = s5[s5["Name of REIT"] == selected_entity].copy()
        if selected_fy != "All":
            df5 = df5[df5["Financial Year"] == selected_fy]
            
        if not df5.empty:
            val_txn_col = "Value of Transaction"
            val1_col = "Valuation 1 (INR Crores)"
            val2_col = "Valuation 2 (INR Crores)"
            
            if all(c in df5.columns for c in [val_txn_col, val1_col, val2_col]):
                def check_valuation(row):
                    txn = clean_currency(row[val_txn_col])
                    v1 = clean_currency(row[val1_col])
                    v2 = clean_currency(row[val2_col])
                    
                    if v1 == 0 and v2 == 0: return "⚠️ Missing Valuations"
                    
                    avg_val = (v1 + v2) / 2
                    limit = 1.10 * avg_val
                    
                    if txn <= limit:
                        return "🟢 Compliant"
                    else:
                        return f"🔴 Exceeds Limit (Limit: {limit:,.2f})"

                df5["Status"] = df5.apply(check_valuation, axis=1)
                st.dataframe(df5, use_container_width=True, hide_index=True)
                
                if df5["Status"].str.contains("🔴").any():
                    st.error("Alert: Some transactions exceed the 110% valuation threshold.")
            else:
                st.warning("Required valuation columns not found in Sheet 5.")
        else:
            st.info("No Acquisition/Disposal data found for selection.")

    st.divider()

    # ================= SECTION 6: Disclosures (Sheet 6) =================
    st.subheader("5. Annual Report Disclosures")
    
    if not s6.empty:
        df6 = s6[s6["Name of REIT"] == selected_entity].copy()
        if selected_fy != "All":
            df6 = df6[df6["Financial Year"] == selected_fy]
            
        if not df6.empty:
            st.dataframe(df6, use_container_width=True, hide_index=True)
        else:
            st.info("No disclosure data found.")