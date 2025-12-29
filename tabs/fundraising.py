# tabs/fundraising.py
import altair as alt
import pandas as pd
import streamlit as st

from utils.common import (
    ENT_COL, FY_COL,
    DEFAULT_REIT_FUND_URL, DEFAULT_INVIT_FUND_URL,
    _find_col, _num_series, _standardize_selector_columns, _to_date,
    load_table_url,
)

# --------------------------- Helpers ---------------------------

def _process_fundraising_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers, compute numeric columns, and format date."""
    df.columns = [c.strip() for c in df.columns]
    df = _standardize_selector_columns(df)

    cols = df.columns
    listed_on_col = _find_col(cols, aliases=["Listed on"])
    pp_listed_col = _find_col(cols, aliases=["Public/ Private Listed", "Public/Private Listed"])
    date_col      = _find_col(cols, aliases=["Date of Fund raising", "Date of Fund Raising"])
    type_col      = _find_col(cols, aliases=["Type of Issue"])
    cat_col       = _find_col(cols, aliases=["Category of Fund Raising", "Category"])
    amt_col       = _find_col(cols, aliases=["Amount of Fund Raised", "Amount Raised"])
    units_col     = _find_col(cols, aliases=["No. of Units Issued"])
    unitcap_col   = _find_col(cols, aliases=["Unit Capital at the end of Fund Raising", "Unit Capital at End"])

    df["Amount of Fund Raised (num)"] = _num_series(df, amt_col)
    df["No. of Units Issued (num)"]   = _num_series(df, units_col)
    df["Unit Capital at End (num)"]   = _num_series(df, unitcap_col)

    if date_col:
        df["Date of Fund raising (fmt)"] = df[date_col].apply(_to_date)
        df["__date__"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    else:
        df["Date of Fund raising (fmt)"] = "-"
        df["__date__"] = pd.NaT

    df.attrs["__fund_cols__"] = {
        "listed_on": listed_on_col,
        "public_private": pp_listed_col,
        "date": date_col,
        "type": type_col,
        "category": cat_col,
        "amount_num": "Amount of Fund Raised (num)",
        "units_num": "No. of Units Issued (num)",
        "unitcap_num": "Unit Capital at End (num)",
    }
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_fundraising_url(url: str) -> pd.DataFrame:
    df = load_table_url(url)
    return _process_fundraising_df(df)


def _inject_page_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.0rem; }
          .stMultiSelect [data-baseweb="tag"] {
            background: #eef2ff !important;
            border: 1px solid #c7d2fe !important;
            color: #1f2937 !important;
            border-radius: 0.5rem !important;
          }
          .stMultiSelect [data-baseweb="tag"] svg { fill: #1f2937 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def multiselect_with_select_all(
    label: str,
    options: list[str],
    *,
    key: str,
    default_all: bool = True,
    help: str | None = None,
) -> list[str]:
    """
    Excel-like multiselect with a built-in 'Select all' *inside* the dropdown.
    """
    sentinel = "Select all"
    display_options = [sentinel] + options

    ss_key = f"{key}_values"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = options[:] if default_all else []

    def _on_change():
        sel = st.session_state.get(ss_key, [])
        if sentinel in sel:
            st.session_state[ss_key] = options[:]

    st.multiselect(
        label,
        options=display_options,
        # The 'default' parameter is removed here to resolve the warning
        key=ss_key,
        help=help,
        on_change=_on_change,
    )

    return [v for v in st.session_state[ss_key] if v in options]

# --------------------------- Page ---------------------------

def render():
    st.header("Fund Raising")
    _inject_page_css()
    with st.sidebar:
        segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_fund")
        # compute the default URL after segment is chosen
        default_url = DEFAULT_INVIT_FUND_URL if segment == "InvIT" else DEFAULT_REIT_FUND_URL

        data_url = st.text_input(
            "Data URL (public Google Sheet / CSV / XLSX / JSON / HTML table)",
            value=default_url,
            key=f"fund_url_{segment}",
        )
        data_url = data_url.strip()

    if not data_url:
        st.warning("Please provide a data URL.")
        st.stop()

    try:
        df = load_fundraising_url(data_url)
    except Exception as e:
        st.error(f"Could not read the URL. Make sure it’s publicly accessible.\n\nDetails: {e}")
        st.stop()

    cols_meta   = df.attrs["__fund_cols__"]
    type_col    = cols_meta["type"]
    cat_col     = cols_meta["category"]

    entities = sorted(df[ENT_COL].dropna().astype(str).unique())
    fy_vals  = sorted(df[FY_COL].dropna().astype(str).unique())
    types    = sorted(df[type_col].dropna().astype(str).unique()) if type_col else []
    cats     = sorted(df[cat_col].dropna().astype(str).unique()) if cat_col else []

    c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.2], gap="small")

    with c1:
        ent_sel = multiselect_with_select_all(
            "Entity", entities, key="fund_ent", default_all=False,
            help="Use “Select all” at the top to include every entity.",
        )
    with c2:
        fy_sel = multiselect_with_select_all("Financial Year", fy_vals, key="fund_fy", default_all=False)
    with c3:
        type_sel = multiselect_with_select_all("Type of Issue", types, key="fund_type", default_all=False)
    with c4:
        cat_sel = multiselect_with_select_all("Category", cats, key="fund_cat", default_all=False)

    if not any([ent_sel, fy_sel, type_sel, cat_sel]):
        st.info("Please make a selection from the filters above to view the dashboard.")
        st.stop()
    
    mask = pd.Series(True, index=df.index)
    if ent_sel: mask &= df[ENT_COL].astype(str).isin(ent_sel)
    if fy_sel: mask &= df[FY_COL].astype(str).isin(fy_sel)
    if type_sel and type_col: mask &= df[type_col].astype(str).isin(type_sel)
    if cat_sel and cat_col: mask &= df[cat_col].astype(str).isin(cat_sel)

    fdf = df[mask].copy()
    if fdf.empty:
        st.warning("No rows match your current filter selection.")
        st.stop()

    total_amount = fdf["Amount of Fund Raised (num)"].sum(skipna=True)
    num_raises   = int(fdf.shape[0])

    k1, k2 = st.columns(2)
    with k1:
        st.metric("Total Amount Raised", f"{total_amount:,.2f}")
    with k2:
        st.metric("Number of Times Fund Raised", f"{num_raises}")
    st.markdown("---")

    palette_fy  = ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"]
    palette_ent = ["#86efac", "#4ade80", "#22c55e", "#16a34a", "#15803d"]

    fy_chart_df = fdf.groupby(FY_COL, as_index=False)["Amount of Fund Raised (num)"].sum()
    if not fy_chart_df.empty:
        c_fy = (alt.Chart(fy_chart_df).mark_bar(size=22, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X(f"{FY_COL}:N", sort=None, title="Financial Year"),
                y=alt.Y("Amount of Fund Raised (num):Q", title="Amount (units as in data)"),
                color=alt.Color(f"{FY_COL}:N", legend=None, scale=alt.Scale(range=palette_fy)),
                tooltip=[alt.Tooltip(FY_COL, title="FY"), alt.Tooltip("Amount of Fund Raised (num)", title="Amount", format=",")],
            ).properties(height=300).configure_axis(grid=True, gridOpacity=0.12))
        st.altair_chart(c_fy, use_container_width=True)

    ent_chart_df = (fdf.groupby(ENT_COL, as_index=False)["Amount of Fund Raised (num)"].sum()
        .sort_values("Amount of Fund Raised (num)", ascending=False).head(12))
    if not ent_chart_df.empty:
        c_ent = (alt.Chart(ent_chart_df).mark_bar(size=18, cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("Amount of Fund Raised (num):Q", title="Amount (units as in data)"),
                y=alt.Y(f"{ENT_COL}:N", sort='-x', title="Entity"),
                color=alt.Color(f"{ENT_COL}:N", legend=None, scale=alt.Scale(range=palette_ent)),
                tooltip=[alt.Tooltip(ENT_COL, title="Entity"), alt.Tooltip("Amount of Fund Raised (num)", title="Amount", format=",")],
            ).properties(height=360).configure_axis(grid=True, gridOpacity=0.12))
        st.altair_chart(c_ent, use_container_width=True)
    st.markdown("---")

    st.markdown("### Records")
    st.dataframe(fdf, use_container_width=True, hide_index=True)

    export_df = fdf.rename(columns={ENT_COL: "Entity", FY_COL: "Financial Year"})
    st.download_button("Download filtered data (CSV)", data=export_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{segment.lower()}_fund_raising_filtered.csv", mime="text/csv")