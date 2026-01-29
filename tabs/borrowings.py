# tabs/borrowings.py
import pandas as pd
import streamlit as st
import altair as alt

from utils.common import (
    DEFAULT_REIT_BORR_URL,
    DEFAULT_INVIT_BORR_URL,
    ENT_COL, FY_COL, QTR_COL,
    load_table_url,
    _standardize_selector_columns,
    _quarter_sort,
    _find_col,
    _num_series,
    inject_global_css
)

# --------------------------- Helpers ---------------------------

def _process_borrowings_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns, ensure numeric types for metrics.
    """
    df.columns = [str(c).strip() for c in df.columns]
    df = _standardize_selector_columns(df)

    # Identify numeric columns using aliases
    cols = df.columns
    borr_col = _find_col(cols, aliases=["Consolidated Borrowings", "Borrowings"])
    def_col  = _find_col(cols, aliases=["Deferred Payments", "Deferred Payment"])
    cash_col = _find_col(cols, aliases=["Cash and Cash Equivalents", "Cash & Cash Equivalents"])
    val_col  = _find_col(cols, aliases=["Value of REIT Assets", "REIT Assets Value", "Asset Value"])
    nbr_col  = _find_col(cols, aliases=["Net Borrowings Ratio (NBR)", "Net Borrowing Ratio"])

    # Create normalized numeric columns
    df["Borrowings (num)"]        = _num_series(df, borr_col)
    df["Deferred Payments (num)"] = _num_series(df, def_col, fill=0.0)
    df["Cash (num)"]              = _num_series(df, cash_col, fill=0.0)
    df["Asset Value (num)"]       = _num_series(df, val_col)
    
    # NBR is often a string "14.83%". Convert to float 14.83
    if nbr_col:
        # Remove % and convert
        df["NBR (num)"] = pd.to_numeric(
            df[nbr_col].astype(str).str.replace("%", "", regex=False), 
            errors='coerce'
        )
    else:
        df["NBR (num)"] = pd.Series([float('nan')] * len(df))

    # Store original column names for display
    df.attrs["__borr_cols__"] = {
        "borr": borr_col,
        "def": def_col,
        "cash": cash_col,
        "val": val_col,
        "nbr": nbr_col,
        "rating1": _find_col(cols, aliases=["Credit Rating CRA1", "Rating 1"]),
        "agency1": _find_col(cols, aliases=["Name of CRA1", "Agency 1"]),
        "rating2": _find_col(cols, aliases=["Credit Rating CRA2", "Rating 2"]),
        "agency2": _find_col(cols, aliases=["Name of CRA2", "Agency 2"]),
    }
    return df

@st.cache_data(show_spinner=False, ttl=300)
def load_borrowings_data(url: str) -> pd.DataFrame:
    df = load_table_url(url)
    return _process_borrowings_df(df)

def render_borrow():
    st.header("Borrowings")
    inject_global_css()

    # --- Sidebar: Segment Selection ---
    with st.sidebar:
        segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_borr")
    
    # Auto-select URL (Hidden from UI)
    data_url = DEFAULT_INVIT_BORR_URL if segment == "InvIT" else DEFAULT_REIT_BORR_URL

    if not data_url:
        st.warning("Data URL not configured.")
        return

    try:
        df = load_borrowings_data(data_url)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if df.empty:
        st.warning("Data loaded but empty.")
        return

    # --- Sidebar: Filters (Single Select) ---
    with st.sidebar:
        st.divider()
        
        # 1. Entity (Single Select)
        entities = sorted(df[ENT_COL].dropna().astype(str).unique())
        if not entities:
            st.warning("No entities found.")
            return
            
        entity = st.selectbox("Entity", entities, key="borr_ent")

        # 2. Financial Year (Dependent on Entity)
        # Filter df temporarily to find relevant FYs
        ent_mask = df[ENT_COL] == entity
        available_fys = sorted(df.loc[ent_mask, FY_COL].dropna().astype(str).unique())
        
        fy = st.selectbox("Financial Year", ["All"] + available_fys, key="borr_fy")

        # 3. Quarter (Dependent on Entity + FY)
        if fy != "All":
            fy_mask = ent_mask & (df[FY_COL].astype(str) == fy)
            available_qtrs = _quarter_sort(df.loc[fy_mask, QTR_COL].dropna().astype(str).unique())
        else:
            # If All FYs selected, show all quarters available for this entity? 
            # Or just "All"? Let's allow filtering by Quarter across years if desired, 
            # or just reset to All. Usually specific Qtr makes sense only with specific FY.
            # We'll allow All or specific quarters present in the whole dataset for this entity.
            available_qtrs = _quarter_sort(df.loc[ent_mask, QTR_COL].dropna().astype(str).unique())

        qtr = st.selectbox("Quarter", ["All"] + available_qtrs, key="borr_qtr")

    # --- Main Content Filtering ---
    # Start with Entity mask
    mask = (df[ENT_COL] == entity)

    # Apply FY filter
    if fy != "All":
        mask &= (df[FY_COL].astype(str) == fy)

    # Apply Quarter filter
    if qtr != "All":
        mask &= (df[QTR_COL].astype(str) == qtr)

    row_df = df[mask].copy()

    if row_df.empty:
        st.info(f"No data found for {entity} matching the selected filters.")
        return

    # --- Display Logic ---
    # If a specific row is pinpointed (Single FY + Single Quarter), show detailed metrics
    # Otherwise (All FYs or All Quarters), show a summary table/chart.

    if fy != "All" and qtr != "All" and len(row_df) == 1:
        # --- Detailed View for a Single Quarter ---
        row = row_df.iloc[0]
        
        st.subheader(f"Snapshot: {fy} | {qtr}")
        
        # 1. Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross Borrowings", f"₹ {row.get('Borrowings (num)', 0):,.0f}")
        c2.metric("Cash & Equiv.", f"₹ {row.get('Cash (num)', 0):,.0f}")
        c3.metric("Net Borrowings Ratio", f"{row.get('NBR (num)', 0)}%")
        c4.metric("Asset Value", f"₹ {row.get('Asset Value (num)', 0):,.0f}")

        st.divider()

        # 2. Credit Ratings
        st.markdown("##### Credit Ratings")
        cols_map = df.attrs["__borr_cols__"]
        
        # Extract ratings safely
        r1 = row.get(cols_map["rating1"], "-")
        a1 = row.get(cols_map["agency1"], "-")
        r2 = row.get(cols_map["rating2"], "-")
        a2 = row.get(cols_map["agency2"], "-")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.info(f"**{a1}**\n\n{r1}")
        with rc2:
            if str(a2).strip() not in ["-", "nan", "None", ""]:
                st.info(f"**{a2}**\n\n{r2}")
            else:
                st.caption("No second rating agency listed.")

    else:
        # --- Trend View (Multiple rows) ---
        st.subheader(f"Trends: {entity}")
        
        # Chart: Borrowings vs Asset Value over time
        # Create a label for the X-axis combining FY and Qtr for sorting
        chart_df = row_df.copy()
        chart_df["Period"] = chart_df[FY_COL].astype(str) + " - " + chart_df[QTR_COL].astype(str)
        
        # Simple Bar Chart for Borrowings
        chart = (
            alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Period", sort=None, title="Period"),
                y=alt.Y("Borrowings (num)", title="Borrowings"),
                tooltip=[FY_COL, QTR_COL, "Borrowings (num)", "NBR (num)"]
            ).properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

        # Data Table
        st.markdown("##### Data Records")
        # Select clean columns to display
        display_cols = [FY_COL, QTR_COL]
        # Add dynamic mapped cols
        cmap = df.attrs["__borr_cols__"]
        for k in ["borr", "cash", "val", "nbr"]:
            if cmap[k] and cmap[k] in row_df.columns:
                display_cols.append(cmap[k])
        
        st.dataframe(row_df[display_cols], use_container_width=True, hide_index=True)