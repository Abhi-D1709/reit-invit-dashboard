# tabs/trading.py
import json
import re
import datetime as dt
import io
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests  # Reverted to standard requests
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit.runtime.scriptrunner import add_script_run_ctx

from utils.common import ENTITIES_SHEET_CSV

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

# --------------------------- live fetchers ---------------------------
@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_bse_data(scripcode: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Standard requests fetcher for BSE (Restored to working version)"""
    scripcode = _normalize_bse_code(scripcode or "")
    if not scripcode: return pd.DataFrame(columns=["date", "close", "volume"])
    
    url = f"https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w?scripcode={scripcode}&flag=1&fromdate={start.strftime('%Y%m%d')}&todate={end.strftime('%Y%m%d')}&seriesid="
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com/"}
    
    try:
        with requests.Session() as s:
            s.get("https://www.bseindia.com/", timeout=10) # Handshake
            r = s.get(url, headers=headers, timeout=10)
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
            
    except Exception:
        return pd.DataFrame(columns=["date", "close", "volume"])

@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_nse_data(symbol: str, series: str, start: dt.date, end: dt.date, user_cookie: str = "") -> pd.DataFrame:
    """
    Fetches NSE data. Uses user-provided cookie to bypass 403 Forbidden errors.
    """
    if start > end: return pd.DataFrame(columns=["date", "close", "volume"])

    # Base Headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "*/*"
    }

    # INJECT USER COOKIE IF PROVIDED
    if user_cookie.strip():
        # Minimal cleaning if they pasted a full curl command header like "Cookie: xxxx"
        clean_cookie = user_cookie
        if "Cookie:" in clean_cookie:
            clean_cookie = clean_cookie.split("Cookie:", 1)[1].strip()
        headers["Cookie"] = clean_cookie

    session = requests.Session()
    session.headers.update(headers)

    def fetch_chunk(d1, d2):
        url = f"https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?functionName=getHistoricalTradeData&symbol={symbol}&series={series}&fromDate={d1.strftime('%d-%m-%Y')}&toDate={d2.strftime('%d-%m-%Y')}&csv=true"
        
        try:
            r = session.get(url, timeout=10)
            
            if r.status_code == 403:
                # If 403, it means the cookie is missing or expired.
                return pd.DataFrame()
            
            r.raise_for_status()
            csv_text = r.text
            if not csv_text or "Date" not in csv_text: return pd.DataFrame()

            df = pd.read_csv(io.StringIO(csv_text))
            df.columns = [c.strip().replace('"', '') for c in df.columns]
            col_map = {c.lower(): c for c in df.columns}
            
            if 'date' not in col_map: return pd.DataFrame()

            df["date"] = pd.to_datetime(df[col_map['date']], errors="coerce").dt.date
            df["close"] = pd.to_numeric(df[col_map['close']].astype(str).str.replace(',', ''), errors="coerce")
            df["volume"] = pd.to_numeric(df[col_map['volume']].astype(str).str.replace(',', ''), errors="coerce")
            
            return df[["date", "close", "volume"]].dropna()

        except Exception:
            return pd.DataFrame()

    # Chunking logic (NSE allows max 365 days)
    CHUNK_DAYS = 360
    windows = []
    current_end = end
    while current_end >= start:
        current_start = max(start, current_end - dt.timedelta(days=CHUNK_DAYS))
        windows.append((current_start, current_end))
        current_end = current_start - dt.timedelta(days=1)
    
    parts = []
    for s_date, e_date in windows:
        p = fetch_chunk(s_date, e_date)
        if not p.empty: parts.append(p)
    
    if not parts: 
        return pd.DataFrame(columns=["date", "close", "volume"])
    
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


# -------------------------- one-entity fetch --------------------------
def fetch_single_entity(
    row: pd.Series, start: dt.date, end: dt.date, monthly_mode: bool, nse_cookie: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    sym, series, scrip = row.get("NSE Symbol", ""), row.get("NSE Series", ""), row.get("BSE Scrip Code", "")
    
    if sym and series:
        # Pass the cookie specifically to NSE fetcher
        nse_df = get_nse_data(sym, series, start, end, user_cookie=nse_cookie)
    else:
        nse_df = pd.DataFrame(columns=["date", "close", "volume"])
        
    if scrip:
        bse_df = get_bse_data(scrip, start, end)
    else:
        bse_df = pd.DataFrame(columns=["date", "close", "volume"])

    if monthly_mode:
        if not nse_df.empty: nse_df = to_monthly(nse_df)
        if not bse_df.empty: bse_df = to_monthly(bse_df)
    
    for df in (nse_df, bse_df):
        if not df.empty:
            df["Entity"], df["Type"] = row["Name of Entity"], row["Type of Entity"]
            
    return nse_df, bse_df
    
# -------------------------- UI Components --------------------------
def render_sidebar(entities_df: pd.DataFrame):
    st.markdown("### Trading — Controls")
    
    # --- New Settings Box for Manual Cookie ---
    with st.expander("⚙️ Connection Settings (NSE)", expanded=False):
        st.caption("If NSE data fails (403), paste the 'Cookie' header from your browser network tab here.")
        nse_cookie = st.text_area("NSE Cookie String", height=100, help="Paste the value of the 'cookie' header from a valid request to nseindia.com")

    start, end = clamp_dates(
        st.date_input("From", value=dt.date(2024, 4, 1), format="DD/MM/YYYY", key="trade_from"),
        st.date_input("To", value=dt.date.today(), format="DD/MM/YYYY", key="trade_to")
    )
    mode = st.radio("Mode", ["Single Entity", "All REITs", "All InvITs"], key="trade_mode", horizontal=True)
    monthly_mode = st.checkbox("Monthly aggregation", value=(mode != "Single Entity"), help="volume=sum, close=last day of month", key="trade_monthly", disabled=(mode != "Single Entity"))
    
    entity_name = None
    if mode == "Single Entity":
        if not entities_df.empty:
            entity_name = st.selectbox("Select Entity", entities_df["Name of Entity"].tolist(), index=0)
    
    return mode, start, end, (monthly_mode or mode != "Single Entity"), entity_name, nse_cookie

def render_single_entity_view(row, start_date, end_date, monthly_mode, nse_cookie):
    st.subheader(f"Entity View: {row['Name of Entity']}")
    nse_df, bse_df = fetch_single_entity(row, start_date, end_date, monthly_mode, nse_cookie)
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        title = f"NSE • {row.get('NSE Symbol', '')}" + (" (Monthly)" if monthly_mode else "")
        if nse_df is None or nse_df.empty: 
            st.warning("NSE: No data for this period.")
            if not nse_cookie: st.info("ℹ️ Try pasting a valid Cookie in 'Connection Settings' in the sidebar.")
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
        # Now returns nse_cookie as well
        mode, start_date, end_date, monthly_mode, entity_name, nse_cookie = render_sidebar(entities_all)

    if entities_all.empty:
        st.error("Could not load entities list. Please check the CSV URL in the sidebar."); return

    if mode == "Single Entity":
        if not entity_name:
            st.info("Select an entity from the sidebar to begin."); return
        selected_row = entities_all[entities_all["Name of Entity"] == entity_name].iloc[0]
        render_single_entity_view(selected_row, start_date, end_date, monthly_mode, nse_cookie)
    
    else: # Group View
        group_type = "REIT" if mode == "All REITs" else "InvIT"
        st.subheader(f"Group View • All {group_type}s")
        rows_to_fetch = entities_all[entities_all["Type of Entity"].str.upper() == group_type.upper()]
        
        if rows_to_fetch.empty:
            st.warning(f"No {group_type}s found in the entities sheet."); return

        daily_nse_parts, daily_bse_parts = [], []
        with st.status(f"Fetching daily data for {len(rows_to_fetch)} {group_type}s...", expanded=True) as status:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_map = {}
                for _, r in rows_to_fetch.iterrows():
                    # Pass the cookie here as well
                    f = executor.submit(fetch_single_entity, r, start_date, end_date, False, nse_cookie)
                    add_script_run_ctx(f)
                    future_map[f] = r

                for future in as_completed(future_map):
                    entity_name = future_map[future]['Name of Entity']
                    try:
                        nse_df, bse_df = future.result()
                        if not nse_df.empty: daily_nse_parts.append(nse_df)
                        if not bse_df.empty: daily_bse_parts.append(bse_df)
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