# tabs/trading.py
"""Trading (NSE & BSE) for tracked REITs/InvITs.

Read-only: prices and volumes come from the Parquet files that jobs/ingest_trades.py
writes to the `data` branch (see utils/datastore.py). This page never fetches from
the exchanges and holds no credentials.
"""
import datetime as dt
import re
from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.common import ENTITIES_SHEET_CSV
from utils.datastore import DataUnavailable, load_manifest, load_trades, source_label

DAILY_COLS = ["date", "close", "vwap", "volume", "turnover", "trades"]


# --------------------------- entities ---------------------------
def _clean_str(x):
    return str(x).strip() if pd.notna(x) else ""


def _normalize_bse_code(s: str) -> str:
    s = _clean_str(s)
    if not s:
        return ""
    if m := re.match(r"^\s*(\d+)(?:\.0+)?\s*$", s):
        return m.group(1)
    return re.sub(r"\D", "", s)


@st.cache_data(ttl=60 * 30, show_spinner="Loading entity list...")
def load_entities(url: str) -> pd.DataFrame:
    cols = ["Type of Entity", "Name of Entity", "NSE Symbol", "NSE Series", "BSE Scrip Code", "ISIN"]
    try:
        df = pd.read_csv(url, dtype=str)
    except Exception as e:
        st.error(f"Failed to load Google Sheet: {e}")
        return pd.DataFrame(columns=cols)

    df = df[[c for c in cols if c in df.columns]].copy()
    for col, func in {
        "Type of Entity": _clean_str,
        "Name of Entity": _clean_str,
        "NSE Symbol": lambda s: _clean_str(s).upper(),
        "NSE Series": lambda s: _clean_str(s).upper(),
        "BSE Scrip Code": _normalize_bse_code,
        "ISIN": _clean_str,
    }.items():
        if col in df.columns:
            df[col] = df[col].map(func)
    return df


def nse_key(row: pd.Series) -> str:
    sym, ser = row.get("NSE Symbol", ""), row.get("NSE Series", "")
    return f"NSE:{sym}:{ser}".upper() if sym and ser else ""


def bse_key(row: pd.Series) -> str:
    scrip = row.get("BSE Scrip Code", "")
    return f"BSE:{scrip}" if scrip else ""


# --------------------------- aggregation ---------------------------
def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Month-end close, summed volume/turnover/trades, and a true monthly VWAP."""
    if df.empty:
        return df
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values("date")
    t["ym"] = t["date"].dt.to_period("M")
    g = t.groupby("ym").agg(
        date=("date", "last"),
        close=("close", "last"),
        volume=("volume", "sum"),
        turnover=("turnover", "sum"),
        trades=("trades", "sum"),
    )
    g["vwap"] = (g["turnover"] / g["volume"]).where(g["volume"] > 0)
    g["date"] = g["date"].dt.date
    return g.reset_index(drop=True)[DAILY_COLS]


def aggregate_volume_and_turnover(df: pd.DataFrame, monthly: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "volume", "turnover"])
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])
    for c in ("volume", "turnover"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    if monthly:
        t["ym"] = t["date"].dt.to_period("M")
        g = t.groupby("ym", as_index=False).agg(volume=("volume", "sum"), turnover=("turnover", "sum"))
        g["date"] = g["ym"].dt.to_timestamp("M").dt.date
    else:
        t["date"] = t["date"].dt.date
        g = t.groupby("date", as_index=False).agg(volume=("volume", "sum"), turnover=("turnover", "sum"))
    return g[["date", "volume", "turnover"]].sort_values("date")


# --------------------------- charts ---------------------------
def line_bar_figure(df: pd.DataFrame, title: str, *, monthly=False) -> Optional[go.Figure]:
    if df.empty:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.5, marker_line_width=0)
    fig.add_trace(go.Scatter(x=df["date"], y=df["vwap"], name="VWAP", mode="lines", line=dict(width=2)), secondary_y=True)
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["close"], name="Close", mode="lines", line=dict(width=1, dash="dot")),
        secondary_y=True,
    )
    fig.update_layout(
        title=title, height=520, barmode="overlay", bargap=0.25 if monthly else 0.10, hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="simple_white",
    )
    if not monthly:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Price (₹)", secondary_y=True)
    return fig


def volume_only_bar(df: pd.DataFrame, title: str, *, monthly=False) -> Optional[go.Figure]:
    if df.empty:
        return None
    fig = go.Figure()
    fig.add_bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.75, marker_line_width=0)
    fig.update_layout(
        title=title, height=420, hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="simple_white", bargap=0.25 if monthly else 0.10,
    )
    if not monthly:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Volume")
    return fig


def clamp_dates(start: dt.date, end: dt.date) -> Tuple[dt.date, dt.date]:
    today = dt.date.today()
    if end > today:
        end = today
    if start > end:
        start = end
    return start, end


# --------------------------- UI ---------------------------
def render_sidebar(entities_df: pd.DataFrame):
    st.markdown("### Trading — Controls")
    start, end = clamp_dates(
        st.date_input("From", value=dt.date(2024, 4, 1), format="DD/MM/YYYY", key="trade_from"),
        st.date_input("To", value=dt.date.today(), format="DD/MM/YYYY", key="trade_to"),
    )
    mode = st.radio("Mode", ["Single Entity", "All REITs", "All InvITs"], key="trade_mode", horizontal=True)
    monthly_mode = st.checkbox(
        "Monthly aggregation", value=(mode != "Single Entity"), key="trade_monthly",
        help="volume/turnover = sum, close = last day of month, VWAP = turnover ÷ volume",
        disabled=(mode != "Single Entity"),
    )
    entity_name = None
    if mode == "Single Entity" and not entities_df.empty:
        entity_name = st.selectbox("Select Entity", entities_df["Name of Entity"].tolist(), index=0)
    return mode, start, end, (monthly_mode or mode != "Single Entity"), entity_name


def _split_frame(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if not key:
        return pd.DataFrame(columns=DAILY_COLS)
    return df[df["key"] == key][DAILY_COLS].reset_index(drop=True)


def render_single_entity_view(row, df, start_date, end_date, monthly_mode):
    st.subheader(f"Entity View: {row['Name of Entity']}")
    nse_df, bse_df = _split_frame(df, nse_key(row)), _split_frame(df, bse_key(row))
    if monthly_mode:
        nse_df, bse_df = to_monthly(nse_df), to_monthly(bse_df)

    c1, c2 = st.columns(2, gap="large")
    for col, label, code, frame, has_listing in (
        (c1, "NSE", row.get("NSE Symbol", ""), nse_df, bool(nse_key(row))),
        (c2, "BSE", row.get("BSE Scrip Code", ""), bse_df, bool(bse_key(row))),
    ):
        with col:
            if not has_listing:
                st.info(f"{label}: not listed / no {label} code in the entities sheet.")
            elif frame.empty:
                st.warning(f"{label}: no trades in this period.")
            else:
                title = f"{label} • {code}" + (" (Monthly)" if monthly_mode else "")
                st.plotly_chart(line_bar_figure(frame, title, monthly=monthly_mode), width="stretch")
                st.dataframe(frame, width="stretch", hide_index=True)


def render_group_view(entities_all, df, group_type):
    st.subheader(f"Group View • All {group_type}s")
    rows = entities_all[entities_all["Type of Entity"].str.upper() == group_type.upper()]
    if rows.empty:
        st.warning(f"No {group_type}s found in the entities sheet.")
        return

    names = {}
    for _, r in rows.iterrows():
        for k in (nse_key(r), bse_key(r)):
            if k:
                names[k] = r["Name of Entity"]
    data = df[df["key"].isin(names)].copy()
    data["Entity"] = data["key"].map(names)
    data = data.rename(columns={"exchange": "Exchange"})
    nse_all, bse_all = data[data["Exchange"] == "NSE"], data[data["Exchange"] == "BSE"]

    tabs = st.tabs(["Aggregated (BSE+NSE)", "Aggregated (NSE)", "Aggregated (BSE)", "Data per Entity"])
    with tabs[3]:
        st.subheader("Daily Data per Entity")
        if data.empty:
            st.info("No daily data to display.")
        else:
            st.dataframe(
                data[["Entity", "date", "Exchange", "close", "vwap", "volume", "turnover", "trades"]].sort_values(["Entity", "date", "Exchange"]),
                width="stretch", hide_index=True,
            )
    for tab, label, frame in ((tabs[0], "BSE + NSE", data), (tabs[1], "NSE", nse_all), (tabs[2], "BSE", bse_all)):
        with tab:
            st.subheader(f"Aggregated Monthly Volume ({label})")
            if frame.empty:
                st.info("No data to aggregate.")
            else:
                agg = aggregate_volume_and_turnover(frame, monthly=True)
                st.plotly_chart(volume_only_bar(agg, f"Total Monthly Volume ({label}) • All {group_type}s", monthly=True), width="stretch")
                st.dataframe(agg, width="stretch", hide_index=True)


def _render_data_status(manifest: dict) -> None:
    last = manifest["trades"]["last_date"]
    st.caption(f"Data as of **{last}** (updated {manifest['generated_at'][:16].replace('T', ' ')} UTC) · source: {source_label()}")
    issues = list(manifest.get("errors", []))
    if issues:
        st.warning(f"The last data update had {len(issues)} error(s); recent days may be incomplete.")
        with st.expander("Update errors"):
            for e in issues:
                st.code(e)


def render():
    st.header("Trading (NSE & BSE)")

    entities_all = load_entities(ENTITIES_SHEET_CSV)
    with st.sidebar:
        mode, start_date, end_date, monthly_mode, entity_name = render_sidebar(entities_all)

    if entities_all.empty:
        st.error("Could not load the entities list from the Google Sheet.")
        return

    try:
        manifest = load_manifest()
    except (DataUnavailable, FileNotFoundError, KeyError, ValueError) as e:
        st.error(f"Trading data is not available yet: {e}")
        return
    _render_data_status(manifest)

    if mode == "Single Entity":
        if not entity_name:
            st.info("Select an entity from the sidebar to begin.")
            return
        row = entities_all[entities_all["Name of Entity"] == entity_name].iloc[0]
        keys = [k for k in (nse_key(row), bse_key(row)) if k]
        df = load_trades(keys, start_date, end_date)
        render_single_entity_view(row, df, start_date, end_date, monthly_mode)
    else:
        group_type = "REIT" if mode == "All REITs" else "InvIT"
        rows = entities_all[entities_all["Type of Entity"].str.upper() == group_type.upper()]
        keys = [k for _, r in rows.iterrows() for k in (nse_key(r), bse_key(r)) if k]
        df = load_trades(keys, start_date, end_date)
        render_group_view(entities_all, df, group_type)
