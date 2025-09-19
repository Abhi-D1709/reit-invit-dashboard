# tabs/fundraising.py
import altair as alt
import pandas as pd
import streamlit as st
from utils.common import (
    ENT_COL, FY_COL,
    DEFAULT_REIT_FUND_URL, DEFAULT_INVIT_FUND_URL,
    _find_col, _num_series, _standardize_selector_columns, _to_date,
    load_table_url
)

def _process_fundraising_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = _standardize_selector_columns(df)

    cols = df.columns
    listed_on_col   = _find_col(cols, aliases=["Listed on"])
    pp_listed_col   = _find_col(cols, aliases=["Public/ Private Listed"])
    date_col        = _find_col(cols, aliases=["Date of Fund raising", "Date of Fund Raising"])
    type_col        = _find_col(cols, aliases=["Type of Issue"])
    cat_col         = _find_col(cols, aliases=["Category of Fund Raising"])
    amt_col         = _find_col(cols, aliases=["Amount of Fund Raised", "Amount Raised"])
    units_col       = _find_col(cols, aliases=["No. of Units Issued"])
    unitcap_col     = _find_col(cols, aliases=["Unit Capital at the end of Fund Raising", "Unit Capital at End"])

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

def render():
    st.header("Fund Raising")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_fund")
    default_url = DEFAULT_INVIT_FUND_URL if segment == "InvIT" else DEFAULT_REIT_FUND_URL

    st.subheader("Data Source")
    st.caption("Paste a public URL (Google Sheet / CSV / XLSX / JSON / HTML table).")
    data_url = st.text_input("Data URL", value=default_url, key=f"fund_url_{segment}")

    if not data_url.strip():
        st.warning("Please provide a data URL.")
        st.stop()

    try:
        df = load_fundraising_url(data_url.strip())
    except Exception as e:
        st.error(f"Could not read the URL. Make sure it’s publicly accessible.\n\nDetails: {e}")
        st.stop()

    # Filters
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            entities = sorted(df[ENT_COL].dropna().astype(str).unique())
            ent_sel = st.multiselect("Entity", entities, default=entities)
        with c2:
            fy_vals = sorted(df[FY_COL].dropna().astype(str).unique())
            fy_sel = st.multiselect("Financial Year", fy_vals, default=fy_vals)
        with c3:
            type_col = df.attrs["__fund_cols__"]["type"]
            types = sorted(df[type_col].dropna().astype(str).unique()) if type_col else []
            type_sel = st.multiselect("Type of Issue", types, default=types)
        with c4:
            cat_col = df.attrs["__fund_cols__"]["category"]
            cats = sorted(df[cat_col].dropna().astype(str).unique()) if cat_col else []
            cat_sel = st.multiselect("Category", cats, default=cats)

    mask = pd.Series(True, index=df.index)
    if ent_sel:
        mask &= df[ENT_COL].astype(str).isin(ent_sel)
    if fy_sel:
        mask &= df[FY_COL].astype(str).isin(fy_sel)
    if type_sel and df.attrs["__fund_cols__"]["type"]:
        mask &= df[df.attrs["__fund_cols__"]["type"]].astype(str).isin(type_sel)
    if cat_sel and df.attrs["__fund_cols__"]["category"]:
        mask &= df[df.attrs["__fund_cols__"]["category"]].astype(str).isin(cat_sel)

    fdf = df[mask].copy()
    if fdf.empty:
        st.warning("No rows match your filters.")
        st.stop()

    # KPIs
    total_amount = fdf["Amount of Fund Raised (num)"].sum(skipna=True)
    num_raises   = int(fdf.shape[0])
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f'<div class="kpi">💰 <b>Total Amount Raised</b><br><span style="font-size:26px;font-weight:700;">{total_amount:,.2f}</span><br><span class="muted">(units as in data)</span></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi">📈 <b>Number of Times Fund Raised</b><br><span style="font-size:26px;font-weight:700;">{num_raises}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Charts (only FY + Entity)
    fy_chart_df = fdf.groupby(FY_COL, as_index=False)["Amount of Fund Raised (num)"].sum()
    if not fy_chart_df.empty:
        c_fy = (
            alt.Chart(fy_chart_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{FY_COL}:N", sort=None, title="Financial Year"),
                y=alt.Y("Amount of Fund Raised (num):Q", title="Amount (as in data)"),
                tooltip=[alt.Tooltip(FY_COL, title="FY"), alt.Tooltip("Amount of Fund Raised (num)", title="Amount", format=",")]
            )
            .properties(height=280)
        )
        st.altair_chart(c_fy, use_container_width=True)

    ent_chart_df = (
        fdf.groupby(ENT_COL, as_index=False)["Amount of Fund Raised (num)"]
        .sum()
        .sort_values("Amount of Fund Raised (num)", ascending=False)
        .head(10)
    )
    if not ent_chart_df.empty:
        c_ent = (
            alt.Chart(ent_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Amount of Fund Raised (num):Q", title="Amount (as in data)"),
                y=alt.Y(f"{ENT_COL}:N", sort='-x', title="Entity"),
                tooltip=[alt.Tooltip(ENT_COL, title="Entity"), alt.Tooltip("Amount of Fund Raised (num)", title="Amount", format=",")]
            )
            .properties(height=320)
        )
        st.altair_chart(c_ent, use_container_width=True)

    st.markdown("---")

    # Table
    listed_on = df.attrs["__fund_cols__"]["listed_on"]
    pp_listed = df.attrs["__fund_cols__"]["public_private"]
    date_fmt  = "Date of Fund raising (fmt)"
    type_col_ = df.attrs["__fund_cols__"]["type"]
    cat_col_  = df.attrs["__fund_cols__"]["category"]

    show_cols = [ENT_COL, FY_COL, date_fmt]
    for optional in [listed_on, pp_listed, type_col_, cat_col_]:
        if optional and optional not in show_cols:
            show_cols.append(optional)
    show_cols += ["Amount of Fund Raised (num)", "No. of Units Issued (num)", "Unit Capital at End (num)"]

    table_df = fdf[show_cols].rename(columns={
        ENT_COL: "Entity",
        FY_COL: "Financial Year",
        "Amount of Fund Raised (num)": "Amount of Fund Raised",
        "No. of Units Issued (num)": "No. of Units Issued",
        "Unit Capital at End (num)": "Unit Capital at End"
    })

    for colname in ["Amount of Fund Raised", "No. of Units Issued", "Unit Capital at End"]:
        if colname in table_df.columns:
            table_df[colname] = table_df[colname].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

    st.markdown("### Records")
    st.dataframe(table_df, use_container_width=True)

    export_df = fdf.rename(columns={ENT_COL: "Entity", FY_COL: "Financial Year"})
    st.download_button(
        "Download filtered data (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{segment.lower()}_fund_raising_filtered.csv",
        mime="text/csv"
    )
