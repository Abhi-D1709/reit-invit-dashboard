# tabs/trading.py
import json
import re
import time
import random
import datetime as dt
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.common import ENTITIES_SHEET_CSV
from utils.db import (
    sb_upsert_trades, sb_dates_in_range, sb_months_in_range,
    sb_load_range, month_ranges_to_fetch, month_ranges_missing_months,
    sb_healthcheck, sb_max_date_for_key,
)

# --------------------------- utilities ---------------------------
def _clean_str(x):
    return str(x).strip() if pd.notna(x) else ""

def _normalize_bse_code(s: str) -> str:
    s = _clean_str(s)
    if not s: return ""
    if m := re.match(r"^\s*(\d+)(?:\.0+)?\s*$", s): return m.group(1)
    return re.sub(r"\D", "", s)

@st.cache_data(ttl=60 * 30, show_spinner="Loading entity list...")
def load_entities(url: str) -> pd.DataFrame:
    cols = ["Type of Entity", "Name of Entity", "NSE Symbol", "NSE Series", "BSE Scrip Code"]
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load Google Sheet: {e}")
        return pd.DataFrame(columns=cols)

    df = df[[c for c in cols if c in df.columns]].copy()
    for col, func in {
        "Type of Entity": _clean_str, "Name of Entity": _clean_str,
        "NSE Symbol": lambda s: _clean_str(s).upper(), "NSE Series": lambda s: _clean_str(s).upper(),
        "BSE Scrip Code": _normalize_bse_code
    }.items():
        if col in df.columns: df[col] = df[col].map(func)
    return df

def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"])
    t["ym"] = t["date"].dt.to_period("M")
    vol = t.groupby("ym", as_index=False)["volume"].sum()
    last_rows = t.groupby("ym", as_index=False).last(numeric_only=False)[["ym", "date", "close"]]
    m = pd.merge(last_rows, vol, on="ym").sort_values("date")[["date", "close", "volume"]]
    m["date"] = m["date"].dt.date
    return m

def aggregate_volume_and_turnover(df: pd.DataFrame, monthly: bool) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["date", "volume", "turnover"])
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce").dropna()
    t["close"] = pd.to_numeric(t["close"], errors="coerce")
    t["volume"] = pd.to_numeric(t["volume"], errors="coerce")
    t["turnover"] = t["close"] * t["volume"]
    
    if monthly:
        t["ym"] = t["date"].dt.to_period("M")
        g = t.groupby("ym", as_index=False).agg(volume=("volume", "sum"), turnover=("turnover", "sum"))
        g["date"] = g["ym"].dt.to_timestamp("M").dt.date
        out = g[["date", "volume", "turnover"]].sort_values("date")
    else:
        t["date"] = t["date"].dt.date
        g = t.groupby("date", as_index=False).agg(volume=("volume", "sum"), turnover=("turnover", "sum"))
        out = g[["date", "volume", "turnover"]].sort_values("date")
    return out

def line_bar_figure(df: pd.DataFrame, title: str, *, monthly=False) -> Optional[go.Figure]:
    if df.empty: return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.5, marker_line_width=0)
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close", mode="lines", line=dict(width=2)), secondary_y=True)
    fig.update_layout(title=title, height=520, barmode="overlay", bargap=0.25 if monthly else 0.10, hovermode="x unified",
                      margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      template="simple_white")
    if not monthly: fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Close", secondary_y=True)
    return fig

def volume_only_bar(df: pd.DataFrame, title: str, *, monthly=False) -> Optional[go.Figure]:
    if df.empty: return None
    fig = go.Figure()
    fig.add_bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.75, marker_line_width=0)
    fig.update_layout(title=title, height=420, hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      template="simple_white", bargap=0.25 if monthly else 0.10)
    if not monthly: fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Volume")
    return fig

def clamp_dates(start: dt.date, end: dt.date) -> Tuple[dt.date, dt.date]:
    today = dt.date.today()
    if end > today: end = today
    if start > end: start = end
    return start, end

def _tail_windows(start: dt.date, end: dt.date, max_dt: Optional[dt.date], monthly: bool):
    if max_dt is None: return [(start, end)]
    if monthly:
        nm_year, nm_month = (max_dt.year + 1, 1) if max_dt.month == 12 else (max_dt.year, max_dt.month + 1)
        d1 = max(dt.date(nm_year, nm_month, 1), start)
    else:
        d1 = max(max_dt + dt.timedelta(days=1), start)
    return [(d1, end)] if d1 <= end else []

# --------------------------- live fetchers ---------------------------
@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_bse_data(scripcode: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    scripcode = _normalize_bse_code(scripcode or "")
    if not scripcode: return pd.DataFrame(columns=["date", "close", "volume"])
    
    # IMPROVEMENT: Use robust headers for BSE as well
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.bseindia.com/"
    }
    
    url = f"https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w?scripcode={scripcode}&flag=1&fromdate={start.strftime('%Y%m%d')}&todate={end.strftime('%Y%m%d')}&seriesid="
    
    try:
        with requests.Session() as s:
            s.get("https://www.bseindia.com/", headers=headers, timeout=15)
            r = s.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            
            raw_data = r.json().get("Data", "[]")
            series = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            
            if not isinstance(series, list) or not series:
                return pd.DataFrame(columns=["date", "close", "volume"])
            
            df = pd.DataFrame(series)
            if df.empty: return df

            required_cols = ["dttm", "vale1", "vole"]
            if not all(col in df.columns for col in required_cols):
                return pd.DataFrame(columns=["date", "close", "volume"])

            df["date"] = pd.to_datetime(df["dttm"], errors="coerce").dt.date
            df["close"] = pd.to_numeric(df["vale1"], errors="coerce")
            df["volume"] = pd.to_numeric(df["vole"], errors="coerce")
            
            return df[["date", "close", "volume"]].dropna().sort_values("date")
            
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError, TypeError):
        return pd.DataFrame(columns=["date", "close", "volume"])

@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_nse_data(symbol: str, series: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Revised NSE fetcher with session reuse and robust series handling
    to prevent blocking on Streamlit Cloud.
    """
    if start > end: return pd.DataFrame(columns=["date", "close", "volume"])
    
    # 1. Use Realistic Headers to bypass basic bot detection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.nseindia.com/",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9"
    }

    # 2. Check multiple series to be safe (IV, RR, EQ, BE) even if sheet says otherwise.
    #    This prevents missing data if NSE reclassifies the stock.
    target_series = list(set([series, "RR", "IV", "EQ", "BE"]))
    series_json = json.dumps(target_series)

    # 3. Initialize Session ONCE outside the loop
    #    This prevents spamming the home page for cookies every loop iteration.
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
    except requests.RequestException:
        # If we can't hit home page, likely IP blocked or network issue
        return pd.DataFrame(columns=["date", "close", "volume"])

    def fetch_range(d1, d2):
        url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series={series_json}&from={d1.strftime('%d-%m-%Y')}&to={d2.strftime('%d-%m-%Y')}"
        try:
            # Reuse the existing session
            r = session.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            
            data = r.json().get("data", [])
            
            if not isinstance(data, list) or not data:
                return pd.DataFrame(columns=["date", "close", "volume"])
            
            df = pd.DataFrame(data)
            if df.empty: return df
            
            required_cols = ["CH_TIMESTAMP", "CH_CLOSING_PRICE", "CH_TOT_TRADED_QTY"]
            if not all(col in df.columns for col in required_cols):
                return pd.DataFrame(columns=["date", "close", "volume"])
            
            df["date"] = pd.to_datetime(df["CH_TIMESTAMP"], errors="coerce").dt.date
            df["close"] = pd.to_numeric(df["CH_CLOSING_PRICE"], errors="coerce")
            df["volume"] = pd.to_numeric(df["CH_TOT_TRADED_QTY"], errors="coerce")
            
            return df[["date", "close", "volume"]].dropna()
                
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            # Optional: Log error to console for debugging
            # print(f"Error fetching {d1}-{d2}: {e}")
            return pd.DataFrame(columns=["date", "close", "volume"])
            
    # Create windows (365 days max)
    windows, e = [], end
    while e >= start:
        s = max(start, e - dt.timedelta(days=364))
        windows.append((s, e)); e = s - dt.timedelta(days=1)
    
    # Fetch sequentially using the shared session
    parts = [fetch_range(d1, d2) for d1, d2 in windows if d1 <= d2]
    
    valid_parts = [part for part in parts if not part.empty]
    if not valid_parts: return pd.DataFrame(columns=["date", "close", "volume"])
    
    # Concatenate and Drop Duplicates (essential since we request multiple series now)
    return pd.concat(valid_parts, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

# ---------------- Supabase-backed cache -----------------
def _get_data_with_db_cache(
    entity_name: str, key: str, exchange: str,
    start: dt.date, end: dt.date,
    fetch_function: Callable, fetch_args: dict,
    use_cache: bool, coverage: str, tail_only: bool
) -> pd.DataFrame:
    if not use_cache:
        return fetch_function(start=start, end=end, **fetch_args)
    
    todo = _tail_windows(start, end, sb_max_date_for_key(key), monthly=(coverage == "monthly")) if tail_only else list(month_ranges_to_fetch(start, end, sb_dates_in_range(key, start, end)))
    fresh_parts = [df for d1, d2 in todo if not (df := fetch_function(start=d1, end=d2, **fetch_args)).empty]

    if fresh_parts:
        all_new = pd.concat(fresh_parts, ignore_index=True)
        rows = all_new.rename(columns={"date": "dt"})[["dt", "close", "volume"]].copy()
        rows["exchange"] = exchange
        rows["entity"] = entity_name
        rows["k"] = key
        sb_upsert_trades(rows)

    return sb_load_range(key, start, end).rename(columns={"dt": "date"})

def get_nse_data_db_cached(entity_name, symbol, series, start, end, **kwargs) -> pd.DataFrame:
    if not symbol or not series: return pd.DataFrame(columns=["date", "close", "volume"])
    return _get_data_with_db_cache(entity_name=entity_name, key=f"NSE:{symbol}:{series}".upper(), exchange="NSE", start=start, end=end, fetch_function=get_nse_data, fetch_args={"symbol": symbol, "series": series}, **kwargs)

def get_bse_data_db_cached(entity_name, scrip, start, end, **kwargs) -> pd.DataFrame:
    if not scrip: return pd.DataFrame(columns=["date", "close", "volume"])
    return _get_data_with_db_cache(entity_name=entity_name, key=f"BSE:{scrip}", exchange="BSE", start=start, end=end, fetch_function=get_bse_data, fetch_args={"scripcode": scrip}, **kwargs)

# -------------------------- one-entity fetch --------------------------
def fetch_single_entity(
    row: pd.Series, start: dt.date, end: dt.date, monthly_mode: bool, use_cache_now: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cov = "monthly" if monthly_mode else "daily"
    sym, series, scrip = row.get("NSE Symbol", ""), row.get("NSE Series", ""), row.get("BSE Scrip Code", "")
    
    nse_df = get_nse_data_db_cached(row["Name of Entity"], sym, series, start, end, use_cache=use_cache_now, coverage=cov, tail_only=True)
    bse_df = get_bse_data_db_cached(row["Name of Entity"], scrip, start, end, use_cache=use_cache_now, coverage=cov, tail_only=True)

    if monthly_mode:
        if nse_df is not None and not nse_df.empty: nse_df = to_monthly(nse_df)
        if bse_df is not None and not bse_df.empty: bse_df = to_monthly(bse_df)
    
    for df in (nse_df, bse_df):
        if df is not None and not df.empty:
            df["Entity"], df["Type"] = row["Name of Entity"], row["Type of Entity"]
            
    return nse_df, bse_df
    
# -------------------------- UI Components --------------------------
def render_sidebar(entities_df: pd.DataFrame):
    st.markdown("### Trading — Controls")
    start, end = clamp_dates(
        st.date_input("From", value=dt.date(2024, 4, 1), format="DD/MM/YYYY", key="trade_from"),
        st.date_input("To", value=dt.date.today(), format="DD/MM/YYYY", key="trade_to")
    )
    mode = st.radio("Mode", ["Single Entity", "All REITs", "All InvITs"], key="trade_mode", horizontal=True)
    monthly_mode = st.checkbox("Monthly aggregation", value=(mode != "Single Entity"), help="volume=sum, close=last day of month", key="trade_monthly", disabled=(mode != "Single Entity"))
    use_db_cache = st.checkbox("Use cloud cache (Supabase)", value=True, key="trade_use_db")
    
    sb_ok = sb_healthcheck() if use_db_cache else False
    if use_db_cache: st.caption("Supabase: " + ("✅ connected" if sb_ok else "⌐ not reachable"))
    
    entity_name = None
    if mode == "Single Entity":
        if not entities_df.empty:
            entity_name = st.selectbox("Select Entity", entities_df["Name of Entity"].tolist(), index=0)
    
    return mode, start, end, (monthly_mode or mode != "Single Entity"), use_db_cache and sb_ok, entity_name

def render_single_entity_view(row, start_date, end_date, monthly_mode, use_cache_now):
    st.subheader(f"Entity View: {row['Name of Entity']}")
    nse_df, bse_df = fetch_single_entity(row, start_date, end_date, monthly_mode, use_cache_now)
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        title = f"NSE • {row.get('NSE Symbol', '')}" + (" (Monthly)" if monthly_mode else "")
        if nse_df is None or nse_df.empty: st.warning("NSE: No data for this period.")
        else:
            st.plotly_chart(line_bar_figure(nse_df, title, monthly=monthly_mode), use_container_width=True)
            st.dataframe(nse_df, use_container_width=True, hide_index=True)
    with c2:
        title = f"BSE • {row.get('BSE Scrip Code', '')}" + (" (Monthly)" if monthly_mode else "")
        if bse_df is None or bse_df.empty: st.warning("BSE: No data for this period.")
        else:
            st.plotly_chart(line_bar_figure(bse_df, title, monthly=monthly_mode), use_container_width=True)
            st.dataframe(bse_df, use_container_width=True, hide_index=True)

# ------------------------------- Main Render ------------------------------
def render():
    st.header("Trading (NSE & BSE)")
    
    entities_all = load_entities(ENTITIES_SHEET_CSV)
    
    with st.sidebar:
        mode, start_date, end_date, monthly_mode, use_cache_now, entity_name = render_sidebar(entities_all)

    if entities_all.empty:
        st.error("Could not load entities list. Please check the CSV URL in the sidebar."); return

    if mode == "Single Entity":
        if not entity_name:
            st.info("Select an entity from the sidebar to begin."); return
        selected_row = entities_all[entities_all["Name of Entity"] == entity_name].iloc[0]
        render_single_entity_view(selected_row, start_date, end_date, monthly_mode, use_cache_now)
    
    else: # Group View
        group_type = "REIT" if mode == "All REITs" else "InvIT"
        st.subheader(f"Group View • All {group_type}s")
        rows_to_fetch = entities_all[entities_all["Type of Entity"].str.upper() == group_type.upper()]
        
        if rows_to_fetch.empty:
            st.warning(f"No {group_type}s found in the entities sheet."); return

        daily_nse_parts, daily_bse_parts = [], []
        with st.status(f"Fetching daily data for {len(rows_to_fetch)} {group_type}s...", expanded=True) as status:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_map = {executor.submit(fetch_single_entity, r, start_date, end_date, False, use_cache_now): r for _, r in rows_to_fetch.iterrows()}
                for future in as_completed(future_map):
                    entity_name = future_map[future]['Name of Entity']
                    try:
                        nse_df, bse_df = future.result()
                        if nse_df is not None and not nse_df.empty: daily_nse_parts.append(nse_df)
                        if bse_df is not None and not bse_df.empty: daily_bse_parts.append(bse_df)
                        status.write(f"✅ Completed: {entity_name}")
                    except Exception as e:
                        status.write(f"⌐ Failed: {entity_name} ({e})")
            status.update(label="All data loaded!", state="complete", expanded=False)

        daily_nse_all = pd.concat(daily_nse_parts, ignore_index=True) if daily_nse_parts else pd.DataFrame()
        daily_bse_all = pd.concat(daily_bse_parts, ignore_index=True) if daily_bse_parts else pd.DataFrame()
        combined_all_daily = pd.concat([daily_nse_all.assign(Exchange="NSE"), daily_bse_all.assign(Exchange="BSE")], ignore_index=True)

        tabs = st.tabs(["Aggregated (BSE+NSE)", "Aggregated (NSE)", "Aggregated (BSE)", "Data per Entity"])
        
        with tabs[3]:
            st.subheader("Daily Data per Entity")
            if combined_all_daily.empty: st.info("No daily data to display.")
            else: st.dataframe(combined_all_daily.sort_values(["Entity", "date", "Exchange"]), use_container_width=True, hide_index=True)
        
        with tabs[0]:
            st.subheader("Aggregated Monthly Volume (BSE + NSE)")
            if combined_all_daily.empty: st.info("No data to aggregate.")
            else:
                agg_df = aggregate_volume_and_turnover(combined_all_daily, monthly=True)
                st.plotly_chart(volume_only_bar(agg_df, f"Total Monthly Volume (BSE+NSE) • All {group_type}s", monthly=True), use_container_width=True)
                st.dataframe(agg_df, use_container_width=True, hide_index=True)
        with tabs[1]:
            st.subheader("Aggregated Monthly Volume (NSE)")
            if daily_nse_all.empty: st.info("No NSE data to aggregate.")
            else:
                agg_df = aggregate_volume_and_turnover(daily_nse_all, monthly=True)
                st.plotly_chart(volume_only_bar(agg_df, f"Total Monthly Volume (NSE) • All {group_type}s", monthly=True), use_container_width=True)
                st.dataframe(agg_df, use_container_width=True, hide_index=True)
        with tabs[2]:
            st.subheader("Aggregated Monthly Volume (BSE)")
            if daily_bse_all.empty: st.info("No BSE data to aggregate.")
            else:
                agg_df = aggregate_volume_and_turnover(daily_bse_all, monthly=True)
                st.plotly_chart(volume_only_bar(agg_df, f"Total Monthly Volume (BSE) • All {group_type}s", monthly=True), use_container_width=True)
                st.dataframe(agg_df, use_container_width=True, hide_index=True)