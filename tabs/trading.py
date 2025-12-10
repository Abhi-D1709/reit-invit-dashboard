# tabs/trading.py

from __future__ import annotations

import json
import re
import math
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterable, Dict

import pandas as pd
import requests
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, random
from decimal import Decimal, InvalidOperation

# ---- App utils (entities sheet URL, CSS, generic table loader) ----
from utils.common import ENTITIES_SHEET_CSV  # Google Sheet (public CSV) for REIT/InvIT master
# Supabase helpers (optional, auto-disabled if no creds)
import utils.db as db

# ============================================================================
# Basics
# ============================================================================
def _clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def _normalize_bse_code(s: str) -> str:
    """
    Normalize BSE scrip code from Google Sheets:
      - "542602", "542602.0", "5.42602E+05", whitespace, etc.
      - Prefer exact integer if numeric; fallback to digits only.
    """
    s = _clean_str(s)
    if not s:
        return ""
    m = re.match(r"^\s*(\d+)(?:\.0+)?\s*$", s)
    if m:
        return m.group(1)
    try:
        d = Decimal(s)
        if d == d.to_integral():
            return str(int(d))
    except (InvalidOperation, ValueError):
        pass
    return re.sub(r"\D", "", s)

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_entities() -> pd.DataFrame:
    """
    Load the REIT/InvIT mapping from Google Sheets (CSV export).
    """
    cols = ["Type of Entity", "Name of Entity", "NSE Symbol", "NSE Series", "BSE Scrip Code"]
    try:
        df = pd.read_csv(ENTITIES_SHEET_CSV)
    except Exception as e:
        st.error(f"Failed to load Entities Google Sheet: {e}")
        return pd.DataFrame(columns=cols)

    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()

    if "Type of Entity" in df.columns:
        df["Type of Entity"] = df["Type of Entity"].map(_clean_str)
    if "Name of Entity" in df.columns:
        df["Name of Entity"] = df["Name of Entity"].map(_clean_str)
    if "NSE Symbol" in df.columns:
        df["NSE Symbol"] = df["NSE Symbol"].map(lambda s: _clean_str(s).upper())
    if "NSE Series" in df.columns:
        df["NSE Series"] = df["NSE Series"].map(lambda s: _clean_str(s).upper())
    if "BSE Scrip Code" in df.columns:
        df["BSE Scrip Code"] = df["BSE Scrip Code"].map(_normalize_bse_code)

    return df

def clamp_dates(start: dt.date, end: dt.date) -> Tuple[dt.date, dt.date]:
    today = dt.date.today()
    if end > today:
        end = today
    if start > end:
        start = end
    return start, end

# ============================================================================
# Chart helpers
# ============================================================================
def _apply_xaxis(fig: go.Figure, *, hide_weekends: bool, monthly: bool):
    if hide_weekends:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    if monthly:
        # force every month to show, and diagonally label at bar centers
        fig.update_xaxes(
            tickformat="%b %Y",
            dtick="M1",
            ticklabelmode="period",
        )

def _base_fig_layout(fig: go.Figure, title: str, height: int = 520, *, hide_weekends=True, monthly=False):
    fig.update_layout(
        title=title,
        height=height,
        barmode="overlay",
        bargap=0.25 if monthly else 0.10,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="simple_white",
    )
    _apply_xaxis(fig, hide_weekends=hide_weekends, monthly=monthly)
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Close", secondary_y=True)

def line_bar_figure(df: pd.DataFrame, title: str, height: int = 520, *, monthly=False) -> Optional[go.Figure]:
    if df.empty:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=df["date"], y=df["volume"], name="Volume",
        opacity=0.5, marker_line_width=0
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["close"], name="Close", mode="lines",
            line=dict(width=2)
        ),
        secondary_y=True,
    )
    _base_fig_layout(fig, title, height, hide_weekends=not monthly, monthly=monthly)
    if monthly:
        tickvals = pd.to_datetime(df["date"])
        ticktext = [d.strftime("%b %Y") for d in tickvals]
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45,  # diagonal labels under bars
        )
    return fig

def volume_only_bar(df: pd.DataFrame, title: str, height: int = 420, *, monthly=False) -> Optional[go.Figure]:
    if df.empty:
        return None
    fig = go.Figure()
    fig.add_bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.75, marker_line_width=0)
    _apply_xaxis(fig, hide_weekends=not monthly, monthly=monthly)
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="simple_white",
        bargap=0.25 if monthly else 0.10,
    )
    if monthly:
        tickvals = pd.to_datetime(df["date"])
        ticktext = [d.strftime("%b %Y") for d in tickvals]
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45,
        )
    fig.update_yaxes(title_text="Volume")
    return fig

# ============================================================================
# Aggregation
# ============================================================================
def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregation per entity: volume=sum, close=last trading day close, date=that last trading day."""
    if df.empty:
        return df
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"]).sort_values("date")
    t["ym"] = t["date"].dt.to_period("M")

    vol = t.groupby("ym", as_index=False)["volume"].sum()
    last_rows = t.groupby("ym", as_index=False).last()[["ym", "date", "close"]]

    m = pd.merge(last_rows, vol, on="ym")
    m = m.sort_values("date")[["date", "close", "volume"]]
    m["date"] = m["date"].dt.date
    return m

def aggregate_volume_and_turnover(df: pd.DataFrame, monthly: bool) -> pd.DataFrame:
    """
    Aggregate across ALL entities (group view).
    - If monthly=True: one row per month
        * volume = sum(volume)
        * turnover = sum(close * volume)
        * avg_daily_turnover = turnover / 22
        * date = month-end
    - If monthly=False: one row per day (calendar date)
        * volume = sum(volume)
        * turnover = sum(close * volume)
    """
    if df.empty:
        cols = ["date", "volume", "turnover"] + (["avg_daily_turnover"] if monthly else [])
        return pd.DataFrame(columns=cols)

    t = df.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["close"] = pd.to_numeric(t["close"], errors="coerce")
    t["volume"] = pd.to_numeric(t["volume"], errors="coerce")
    t = t.dropna(subset=["date"])
    t["turnover"] = t["close"] * t["volume"]

    if monthly:
        t["ym"] = t["date"].dt.to_period("M")
        g = t.groupby("ym", as_index=False).agg(
            volume=("volume", "sum"),
            turnover=("turnover", "sum"),
        )
        g["date"] = g["ym"].dt.to_timestamp("M").dt.date
        g["avg_daily_turnover"] = g["turnover"] / 22.0
        out = g[["date", "volume", "turnover", "avg_daily_turnover"]].sort_values("date")
    else:
        g = t.groupby(t["date"].dt.date, as_index=False).agg(
            volume=("volume", "sum"),
            turnover=("turnover", "sum"),
        )
        out = g[["date", "volume", "turnover"]].sort_values("date")

    # numeric cleanup
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# ============================================================================
# External APIs (NSE/BSE) — raw calls for a single range
# ============================================================================
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def _nse_fetch_range(symbol: str, series: str, d1: dt.date, d2: dt.date) -> pd.DataFrame:
    """
    One call to NSE historical API for [d1..d2]. Returns date, close, volume.
    """
    series_json = json.dumps([series])
    url = (
        "https://www.nseindia.com/api/historicalOR/cm/equity"
        f"?symbol={symbol}&series={series_json}&from={d1.strftime('%d-%m-%Y')}&to={d2.strftime('%d-%m-%Y')}"
    )
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    }
    last_err = None
    for _ in range(3):
        try:
            s = requests.Session()
            s.headers.update(headers)
            s.get("https://www.nseindia.com", timeout=25)  # cookie warm-up
            r = s.get(url, timeout=25)
            r.raise_for_status()
            j = r.json()
            if j.get("error"):
                msg = str(j.get("showMessage") or "")
                if "No record" in msg:
                    return pd.DataFrame(columns=["date", "close", "volume"])
                raise ValueError(msg or "NSE API error")
            data = j.get("data", [])
            if not data:
                return pd.DataFrame(columns=["date", "close", "volume"])
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["CH_TIMESTAMP"], errors="coerce").dt.date
            df["close"] = pd.to_numeric(df["CH_CLOSING_PRICE"], errors="coerce")
            df["volume"] = pd.to_numeric(df["CH_TOT_TRADED_QTY"], errors="coerce")
            return df[["date", "close", "volume"]].dropna()
        except Exception as e:
            last_err = e
            time.sleep(0.35 + random.random() * 0.4)
    st.warning(f"NSE {_fmt(d1)} → {_fmt(d2)}: {type(last_err).__name__}: {last_err}")
    return pd.DataFrame(columns=["date", "close", "volume"])

def _bse_fetch_range(scripcode: str, d1: dt.date, d2: dt.date) -> pd.DataFrame:
    """
    One call to BSE 'StockReachGraph' for [d1..d2]. Returns date, close, volume.
    Tries flag=1 then flag=0.
    """
    scripcode = _normalize_bse_code(scripcode or "")
    if not scripcode:
        return pd.DataFrame(columns=["date", "close", "volume"])

    def build_url(flag_val: int) -> str:
        return (
            "https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w"
            f"?scripcode={scripcode}"
            f"&flag={flag_val}"
            f"&fromdate={d1.strftime('%Y%m%d')}"
            f"&todate={d2.strftime('%Y%m%d')}"
            "&seriesid="
        )

    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "DNT": "1",
    }

    def parse_json_safely(resp: requests.Response) -> dict:
        text = resp.text.lstrip("\ufeff").strip()
        ctype = (resp.headers.get("content-type") or "").lower()
        if "text/html" in ctype or "<html" in text[:400].lower():
            raise ValueError("BSE API returned HTML (bot-protection/maintenance)")
        try:
            return resp.json()
        except json.JSONDecodeError:
            si, ei = text.find("{"), text.rfind("}")
            if si != -1 and ei != -1 and ei > si:
                return json.loads(text[si:ei+1])
            raise

    def to_df(payload) -> pd.DataFrame:
        raw = payload.get("Data", "[]")
        try:
            if isinstance(raw, str):
                series = json.loads(raw or "[]")
            elif isinstance(raw, list):
                series = raw
            else:
                series = []
        except json.JSONDecodeError:
            series = []
        if not series:
            return pd.DataFrame(columns=["date", "close", "volume"])
        df = pd.DataFrame(series)
        df["date"] = pd.to_datetime(df.get("dttm"), errors="coerce").dt.date
        df["close"] = pd.to_numeric(df.get("vale1"), errors="coerce")
        df["volume"] = pd.to_numeric(df.get("vole"), errors="coerce")
        return df[["date", "close", "volume"]].dropna().sort_values("date")

    last_err = None
    for _ in range(3):
        try:
            sess = requests.Session()
            sess.headers.update(headers)
            sess.get("https://www.bseindia.com/", timeout=25)  # cookie warm-up
            r = sess.get(build_url(flag_val=1), timeout=25)
            r.raise_for_status()
            j = parse_json_safely(r)
            df = to_df(j)
            if not df.empty:
                return df
            # try flag=0
            r2 = sess.get(build_url(flag_val=0), timeout=25)
            r2.raise_for_status()
            j2 = parse_json_safely(r2)
            df2 = to_df(j2)
            if not df2.empty:
                return df2
            time.sleep(0.35 + random.random() * 0.4)
        except Exception as e:
            last_err = e
            time.sleep(0.5 + random.random() * 0.6)

    st.warning(f"BSE fetch failed/silent for scrip {scripcode}. Last error: {type(last_err).__name__ if last_err else 'None'}")
    return pd.DataFrame(columns=["date", "close", "volume"])

def _fmt(d: dt.date) -> str:
    return d.strftime("%d %b %Y")

# ============================================================================
# “Smart” fetch with Supabase cache (optional), with 1-year slicing logic
# ============================================================================
def _year_windows_backward(start: dt.date, end: dt.date) -> List[Tuple[dt.date, dt.date]]:
    """Slice [start..end] into <=1 year windows walking backward from end."""
    if start > end:
        start, end = end, end
    total_days = (end - start).days
    if total_days <= 365:
        return [(start, end)]
    windows: List[Tuple[dt.date, dt.date]] = []
    e = end
    while e >= start:
        s = max(start, e - dt.timedelta(days=364))
        windows.append((s, e))
        e = s - dt.timedelta(days=1)
    return windows

def _ensure_cols(df: pd.DataFrame, exchange: str, key: str, entity: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume", "exchange", "k", "entity"])
    t = df.copy()
    t["exchange"] = exchange
    t["k"] = key
    if entity is not None:
        t["entity"] = entity
    return t

def fetch_nse(symbol: str, series: str, start: dt.date, end: dt.date, *, use_db=True) -> pd.DataFrame:
    """
    Try database-first (fill missing months/days), otherwise fallback to live NSE.
    Always obeys <=1-year API windows.
    Returns df: [date, close, volume]
    """
    symbol = (symbol or "").strip().upper()
    series = (series or "").strip().upper()
    if not symbol or not series:
        return pd.DataFrame(columns=["date", "close", "volume"])

    key = f"NSE:{symbol}:{series}"

    if use_db and db.SB_ENABLED and db.sb_healthcheck():
        # what dates do we already have?
        have_dates = db.sb_dates_in_range(key, start, end)
        # which months/days are missing?
        for d1, d2 in db.month_ranges_to_fetch(start, end, have_dates):
            # even within a month, keep the API rule: max 1 year (months are << 1 year)
            df_new = _nse_fetch_range(symbol, series, d1, d2)
            if not df_new.empty:
                up = df_new.copy()
                up["dt"] = pd.to_datetime(up["date"]).dt.date
                up["exchange"] = "NSE"
                up["entity"] = None
                up["k"] = key
                db.sb_upsert_trades(up[["dt", "close", "volume", "exchange", "entity", "k"]])

        # now read the full requested range from DB
        got = db.sb_load_range(key, start, end)
        if not got.empty:
            got = got.rename(columns={"dt": "date"})
            return got[["date", "close", "volume"]]

        # if DB had nothing, fallback to direct
    # Direct fetch (or DB disabled)
    out_parts: List[pd.DataFrame] = []
    for (d1, d2) in _year_windows_backward(start, end):
        dfp = _nse_fetch_range(symbol, series, d1, d2)
        if not dfp.empty:
            out_parts.append(dfp)
    if not out_parts:
        return pd.DataFrame(columns=["date", "close", "volume"])
    out = pd.concat(out_parts, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    out = out.astype({"close": "float64", "volume": "float64"})
    return out

def fetch_bse(scripcode: str, start: dt.date, end: dt.date, *, use_db=True) -> pd.DataFrame:
    scrip = _normalize_bse_code(scripcode or "")
    if not scrip:
        return pd.DataFrame(columns=["date", "close", "volume"])
    key = f"BSE:{scrip}"

    if use_db and db.SB_ENABLED and db.sb_healthcheck():
        have_dates = db.sb_dates_in_range(key, start, end)
        for d1, d2 in db.month_ranges_to_fetch(start, end, have_dates):
            df_new = _bse_fetch_range(scrip, d1, d2)
            if not df_new.empty:
                up = df_new.copy()
                up["dt"] = pd.to_datetime(up["date"]).dt.date
                up["exchange"] = "BSE"
                up["entity"] = None
                up["k"] = key
                db.sb_upsert_trades(up[["dt", "close", "volume", "exchange", "entity", "k"]])
        got = db.sb_load_range(key, start, end)
        if not got.empty:
            got = got.rename(columns={"dt": "date"})
            return got[["date", "close", "volume"]]

    # fallback direct
    out_parts: List[pd.DataFrame] = []
    for (d1, d2) in _year_windows_backward(start, end):
        dfp = _bse_fetch_range(scrip, d1, d2)
        if not dfp.empty:
            out_parts.append(dfp)
    if not out_parts:
        return pd.DataFrame(columns=["date", "close", "volume"])
    out = pd.concat(out_parts, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    out = out.astype({"close": "float64", "volume": "float64"})
    return out

# ============================================================================
# Multi-entity fetch
# ============================================================================
@dataclass
class EntityRow:
    type_of_entity: str
    name: str
    nse_symbol: str
    nse_series: str
    bse_scrip: str

def _row_to_entity(sr: pd.Series) -> EntityRow:
    return EntityRow(
        type_of_entity=sr.get("Type of Entity", ""),
        name=sr.get("Name of Entity", ""),
        nse_symbol=sr.get("NSE Symbol", ""),
        nse_series=sr.get("NSE Series", ""),
        bse_scrip=sr.get("BSE Scrip Code", ""),
    )

def fetch_single_entity(row: EntityRow, start: dt.date, end: dt.date, monthly_mode: bool, *, use_db=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nse_df = pd.DataFrame(columns=["date", "close", "volume"])
    bse_df = pd.DataFrame(columns=["date", "close", "volume"])

    if row.nse_symbol and row.nse_series:
        try:
            nse_df = fetch_nse(row.nse_symbol, row.nse_series, start, end, use_db=use_db)
            if monthly_mode and not nse_df.empty:
                nse_df = to_monthly(nse_df)
        except Exception as e:
            st.warning(f"NSE fetch failed for {row.name}: {e}")

    if row.bse_scrip:
        try:
            bse_df = fetch_bse(row.bse_scrip, start, end, use_db=use_db)
            if monthly_mode and not bse_df.empty:
                bse_df = to_monthly(bse_df)
        except Exception as e:
            st.warning(f"BSE fetch failed for {row.name}: {e}")

    if not nse_df.empty:
        nse_df["Entity"] = row.name
        nse_df["Type"] = row.type_of_entity
        nse_df["Exchange"] = "NSE"
    if not bse_df.empty:
        bse_df["Entity"] = row.name
        bse_df["Type"] = row.type_of_entity
        bse_df["Exchange"] = "BSE"

    return nse_df, bse_df

def fetch_group(df_entities: pd.DataFrame, entity_type: str, start: dt.date, end: dt.date, monthly_mode: bool, *, use_db=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = df_entities[df_entities["Type of Entity"].str.upper() == entity_type.upper()]
    if rows.empty:
        empty_cols = ["date", "close", "volume", "Entity", "Type", "Exchange"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    nse_parts: List[pd.DataFrame] = []
    bse_parts: List[pd.DataFrame] = []

    # modest parallelism
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = []
        for _, r in rows.iterrows():
            ent = _row_to_entity(r)
            futs.append(ex.submit(fetch_single_entity, ent, start, end, monthly_mode, use_db))
        for fut in as_completed(futs):
            nse_df, bse_df = fut.result()
            if not nse_df.empty:
                nse_parts.append(nse_df)
            if not bse_df.empty:
                bse_parts.append(bse_df)

    def _concat(parts: List[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=["date", "close", "volume", "Entity", "Type", "Exchange"])
        out = pd.concat(parts, ignore_index=True)
        return out.sort_values(["Entity", "date"]).reset_index(drop=True)

    return _concat(nse_parts), _concat(bse_parts)

# ============================================================================
# UI
# ============================================================================
def render():
    st.title("REIT / InvIT • NSE & BSE Price–Volume")

    # Sidebar
    with st.sidebar:
        st.markdown("### Date range")
        default_start = dt.date(2024, 4, 1)
        default_end = dt.date.today()

        start_date = st.date_input("From", value=default_start, format="DD/MM/YYYY")
        end_date = st.date_input("To", value=default_end, format="DD/MM/YYYY")
        start_date, end_date = clamp_dates(start_date, end_date)

        monthly_mode = st.checkbox(
            "Monthly",
            value=False,
            help="Aggregate to monthly: volume=sum, close=last trading day of month. In Group view, bars become one per month."
        )

        st.markdown("---")
        mode = st.radio("Mode", ["Single Entity", "All REITs", "All InvITs"], horizontal=False)

        st.markdown("---")
        use_db = st.checkbox("Use Supabase cache (if configured)", value=True, help="Load existing data from DB and only fetch missing periods from the exchange APIs.")
        show_logs = st.checkbox("Show Supabase log", value=False)
        st.session_state["trade_show_sb_logs"] = bool(show_logs)

        with st.expander("(Optional) Inspect Entity Master"):
            ents_preview = load_entities()
            st.dataframe(ents_preview, use_container_width=True, hide_index=True)

        entity_name = None
        if mode == "Single Entity":
            ents_all = load_entities()
            if ents_all.empty:
                st.error("No entities found in the Google Sheet.")
            else:
                ents_ordered = pd.concat(
                    [ents_all[ents_all["Type of Entity"] == "REIT"],
                     ents_all[ents_all["Type of Entity"] == "InvIT"]],
                    ignore_index=True
                )
                entity_name = st.selectbox("Select Entity", ents_ordered["Name of Entity"].tolist(), index=0)

        go_btn = st.button("Load / Refresh")

    if not go_btn:
        st.info("Choose a mode, pick your dates (and Monthly if needed), then click **Load / Refresh**.")
        return

    df_master = load_entities()
    if df_master.empty:
        st.stop()

    if mode == "Single Entity":
        if not entity_name:
            st.error("Please select an entity in the sidebar.")
            st.stop()

        row = df_master[df_master["Name of Entity"] == entity_name]
        if row.empty:
            st.error("Selected entity not found in the Google Sheet.")
            st.stop()
        er = _row_to_entity(row.iloc[0])

        col1, col2 = st.columns((1, 1), gap="large")

        # NSE
        with st.spinner(f"Fetching NSE: {er.name}"):
            try:
                nse_df = fetch_nse(er.nse_symbol, er.nse_series, start_date, end_date, use_db=use_db) if (er.nse_symbol and er.nse_series) else pd.DataFrame(columns=["date","close","volume"])
                if monthly_mode and not nse_df.empty:
                    nse_df = to_monthly(nse_df)
                if nse_df.empty:
                    st.warning("NSE: No data (symbol/series may be missing or no trading data).")
                else:
                    title = f"NSE • {er.name} • {er.nse_symbol} ({er.nse_series})" + (" • Monthly" if monthly_mode else "")
                    with col1:
                        st.plotly_chart(
                            line_bar_figure(nse_df, title, height=560 if not monthly_mode else 520, monthly=monthly_mode),
                            use_container_width=True,
                            config={"displayModeBar": True, "scrollZoom": True},
                        )
                        st.caption(f"Rows: {len(nse_df)} | Range: {nse_df['date'].min()} → {nse_df['date'].max()}")
                        st.dataframe(nse_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download NSE CSV" + (" (Monthly)" if monthly_mode else ""),
                            nse_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"NSE_{er.nse_symbol}_{er.nse_series}_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.exception(e)

        # BSE
        with st.spinner(f"Fetching BSE: {er.name}"):
            try:
                bse_df = fetch_bse(er.bse_scrip, start_date, end_date, use_db=use_db) if er.bse_scrip else pd.DataFrame(columns=["date","close","volume"])
                if monthly_mode and not bse_df.empty:
                    bse_df = to_monthly(bse_df)
                if bse_df.empty:
                    st.warning("BSE: No data (scrip code may be missing or no trading data).")
                else:
                    title = f"BSE • {er.name} • {er.bse_scrip}" + (" • Monthly" if monthly_mode else "")
                    with col2:
                        st.plotly_chart(
                            line_bar_figure(bse_df, title, height=560 if not monthly_mode else 520, monthly=monthly_mode),
                            use_container_width=True,
                            config={"displayModeBar": True, "scrollZoom": True},
                        )
                        st.caption(f"Rows: {len(bse_df)} | Range: {bse_df['date'].min()} → {bse_df['date'].max()}")
                        st.dataframe(bse_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download BSE CSV" + (" (Monthly)" if monthly_mode else ""),
                            bse_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"BSE_{er.bse_scrip}_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.exception(e)

        if show_logs:
            db.show_log_widget()
        return

    # ---------------- Group views ----------------
    group_type = "REIT" if mode == "All REITs" else "InvIT"
    st.subheader(f"Group View • {group_type}s")

    with st.spinner(f"Fetching all {group_type}s... (NSE & BSE)"):
        nse_all, bse_all = fetch_group(df_master, group_type, start_date, end_date, monthly_mode, use_db=use_db)

    if nse_all.empty and bse_all.empty:
        st.warning("No data returned for the chosen range/group (check symbols/series/scrip codes).")
        if show_logs:
            db.show_log_widget()
        return

    tabs = st.tabs([
        "NSE (per entity)",
        "BSE (per entity)",
        "BSE & NSE (per entity)",
        "Aggregated Volume – NSE",
        "Aggregated Volume – BSE",
        "Aggregated Volume – BSE & NSE",
    ])

    # --- NSE per entity
    with tabs[0]:
        if nse_all.empty:
            st.info("NSE: No data.")
        else:
            st.dataframe(nse_all.sort_values(["Entity","date"]), use_container_width=True, hide_index=True)
            st.download_button(
                "Download NSE (Per Entity)" + (" (Monthly)" if monthly_mode else ""),
                nse_all.to_csv(index=False).encode("utf-8"),
                file_name=f"NSE_{group_type}_PER_ENTITY_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                mime="text/csv",
            )

    # --- BSE per entity
    with tabs[1]:
        if bse_all.empty:
            st.info("BSE: No data.")
        else:
            st.dataframe(bse_all.sort_values(["Entity","date"]), use_container_width=True, hide_index=True)
            st.download_button(
                "Download BSE (Per Entity)" + (" (Monthly)" if monthly_mode else ""),
                bse_all.to_csv(index=False).encode("utf-8"),
                file_name=f"BSE_{group_type}_PER_ENTITY_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                mime="text/csv",
            )

    # --- Combined per entity (NSE+BSE)
    with tabs[2]:
        if nse_all.empty and bse_all.empty:
            st.info("No data.")
        else:
            both_per_entity = pd.concat([nse_all, bse_all], ignore_index=True)
            both_per_entity = both_per_entity.sort_values(["Entity", "Exchange", "date"])
            st.dataframe(both_per_entity, use_container_width=True, hide_index=True)
            st.download_button(
                "Download BSE & NSE (Per Entity)" + (" (Monthly)" if monthly_mode else ""),
                both_per_entity.to_csv(index=False).encode("utf-8"),
                file_name=f"BOTH_{group_type}_PER_ENTITY_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                mime="text/csv",
            )

    # --- Aggregated Volume – NSE
    with tabs[3]:
        if nse_all.empty:
            st.info("NSE: No data for aggregate volume.")
        else:
            nse_agg = aggregate_volume_and_turnover(nse_all, monthly_mode)
            st.plotly_chart(
                volume_only_bar(nse_agg[["date","volume"]], f"NSE • Total Volume • All {group_type}s" + (" • Monthly" if monthly_mode else ""), monthly=monthly_mode),
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True},
            )
            st.dataframe(nse_agg, use_container_width=True, hide_index=True)
            st.download_button(
                "Download NSE Aggregate" + (" (Monthly + Avg Daily Turnover)" if monthly_mode else ""),
                nse_agg.to_csv(index=False).encode("utf-8"),
                file_name=f"NSE_{group_type}_AGG_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                mime="text/csv",
            )

    # --- Aggregated Volume – BSE
    with tabs[4]:
        if bse_all.empty:
            st.info("BSE: No data for aggregate volume.")
        else:
            bse_agg = aggregate_volume_and_turnover(bse_all, monthly_mode)
            st.plotly_chart(
                volume_only_bar(bse_agg[["date","volume"]], f"BSE • Total Volume • All {group_type}s" + (" • Monthly" if monthly_mode else ""), monthly=monthly_mode),
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True},
            )
            st.dataframe(bse_agg, use_container_width=True, hide_index=True)
            st.download_button(
                "Download BSE Aggregate" + (" (Monthly + Avg Daily Turnover)" if monthly_mode else ""),
                bse_agg.to_csv(index=False).encode("utf-8"),
                file_name=f"BSE_{group_type}_AGG_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                mime="text/csv",
            )

    # --- Aggregated Volume – BOTH
    with tabs[5]:
        both_all = pd.concat([nse_all, bse_all], ignore_index=True) if (not nse_all.empty or not bse_all.empty) else pd.DataFrame(columns=["date","close","volume"])
        if both_all.empty:
            st.info("No data for aggregate volume.")
        else:
            both_agg = aggregate_volume_and_turnover(both_all, monthly_mode)
            st.plotly_chart(
                volume_only_bar(both_agg[["date","volume"]], f"BSE & NSE • Total Volume • All {group_type}s" + (" • Monthly" if monthly_mode else ""), monthly=monthly_mode),
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True},
            )
            st.dataframe(both_agg, use_container_width=True, hide_index=True)
            st.download_button(
                "Download BSE & NSE Aggregate" + (" (Monthly + Avg Daily Turnover)" if monthly_mode else ""),
                both_agg.to_csv(index=False).encode("utf-8"),
                file_name=f"BOTH_{group_type}_AGG_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                mime="text/csv",
            )

    if show_logs:
        db.show_log_widget()
