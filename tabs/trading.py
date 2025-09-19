# tabs/trading.py
import json
import re
import time
import random
import datetime as dt
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple, List

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
    if pd.isna(x):
        return ""
    return str(x).strip()

def _normalize_bse_code(s: str) -> str:
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
def load_entities(url: str) -> pd.DataFrame:
    cols = ["Type of Entity", "Name of Entity", "NSE Symbol", "NSE Series", "BSE Scrip Code"]
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load Google Sheet: {e}")
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

def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values("date")
    t["ym"] = t["date"].dt.to_period("M")
    vol = t.groupby("ym", as_index=False)["volume"].sum()
    last_rows = t.groupby("ym", as_index=False).last()[["ym", "date", "close"]]
    m = pd.merge(last_rows, vol, on="ym")
    m = m.sort_values("date")[["date", "close", "volume"]]
    m["date"] = m["date"].dt.date
    return m

def aggregate_volume_and_turnover(df: pd.DataFrame, monthly: bool) -> pd.DataFrame:
    """Robust aggregator that always returns a frame with 'date'."""
    if df.empty:
        return pd.DataFrame(columns=["date", "volume"])
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])
    t["close"] = pd.to_numeric(t["close"], errors="coerce")
    t["volume"] = pd.to_numeric(t["volume"], errors="coerce")
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
        # daily: ensure a real 'date' column exists (not an anonymous group key)
        t["date"] = t["date"].dt.date
        g = t.groupby("date", as_index=False).agg(
            volume=("volume", "sum"),
            turnover=("turnover", "sum"),
        )
        out = g[["date", "volume", "turnover"]].sort_values("date")

    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def _apply_xaxis(fig: go.Figure, *, hide_weekends: bool, monthly: bool):
    if hide_weekends:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    if monthly:
        fig.update_xaxes(tickformat="%b %Y", dtick="M1", ticklabelmode="period", tickangle=0)

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
    fig.add_bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.5, marker_line_width=0)
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close", mode="lines", line=dict(width=2)), secondary_y=True)
    _base_fig_layout(fig, title, height, hide_weekends=not monthly, monthly=monthly)
    if monthly:
        tickvals = pd.to_datetime(df["date"])
        ticktext = [d.strftime("%b %Y") for d in tickvals]
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=45)
    return fig

def volume_only_bar(df: pd.DataFrame, title: str, height: int = 420, *, monthly=False) -> Optional[go.Figure]:
    if df.empty:
        return None
    fig = go.Figure()
    fig.add_bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.75, marker_line_width=0)
    _apply_xaxis(fig, hide_weekends=not monthly, monthly=monthly)
    fig.update_yaxes(title_text="Volume")
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
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=45)
    return fig

def clamp_dates(start: dt.date, end: dt.date) -> Tuple[dt.date, dt.date]:
    today = dt.date.today()
    if end > today:
        end = today
    if start > end:
        start = end
    return start, end

def _tail_windows_daily(start: dt.date, end: dt.date, max_dt: dt.date | None):
    """Fetch only after max_dt (or the whole span if no data yet)."""
    if max_dt is None:
        return [(start, end)]
    d1 = max(max_dt + dt.timedelta(days=1), start)
    return [(d1, end)] if d1 <= end else []

def _tail_windows_monthly(start: dt.date, end: dt.date, max_dt: dt.date | None):
    """For monthly coverage: start from the month AFTER max_dt."""
    if max_dt is None:
        return [(start, end)]
    nm = dt.date(max_dt.year + (1 if max_dt.month == 12 else 0),
                 1 if max_dt.month == 12 else max_dt.month + 1, 1)
    d1 = max(nm, start)
    return [(d1, end)] if d1 <= end else []

# --------------------------- live fetchers ---------------------------

@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_bse_data(scripcode: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    scripcode = _normalize_bse_code(scripcode or "")
    if not scripcode:
        return pd.DataFrame(columns=["date", "close", "volume"]).astype(
            {"date": "datetime64[ns]", "close": "float64", "volume": "float64"}
        )

    def build_url(flag_val: int) -> str:
        return (
            "https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w"
            f"?scripcode={scripcode}&flag={flag_val}"
            f"&fromdate={start.strftime('%Y%m%d')}&todate={end.strftime('%Y%m%d')}"
            "&seriesid="
        )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "DNT": "1",
    }

    def try_fetch(flag_val: int) -> requests.Response:
        sess = requests.Session()
        sess.headers.update(headers)
        sess.get("https://www.bseindia.com/", timeout=25)
        return sess.get(build_url(flag_val), timeout=25)

    def parse_json_safely(resp: requests.Response) -> dict:
        text = resp.text.lstrip("\ufeff").strip()
        ctype = (resp.headers.get("content-type") or "").lower()
        if "text/html" in ctype or "<html" in text[:400].lower():
            raise ValueError("BSE API returned HTML")
        try:
            return resp.json()
        except json.JSONDecodeError:
            si, ei = text.find("{"), text.rfind("}")
            if si != -1 and ei != -1 and ei > si:
                return json.loads(text[si:ei + 1])
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
            r = try_fetch(flag_val=1)
            r.raise_for_status()
            j = parse_json_safely(r)
            df = to_df(j)
            if not df.empty:
                return df
            r2 = try_fetch(flag_val=0)
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
    return pd.DataFrame(columns=["date", "close", "volume"]).astype(
        {"date": "datetime64[ns]", "close": "float64", "volume": "float64"}
    )

@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_nse_data(symbol: str, series: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    if start > end:
        start, end = end, end
    total_days = (end - start).days
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain, */*", "Referer": "https://www.nseindia.com/"}
    series_json = json.dumps([series])

    def empty_df() -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "close", "volume"])

    def fetch_range(d1: dt.date, d2: dt.date) -> pd.DataFrame:
        url = (
            "https://www.nseindia.com/api/historicalOR/cm/equity"
            f"?symbol={symbol}&series={series_json}&from={d1.strftime('%d-%m-%Y')}&to={d2.strftime('%d-%m-%Y')}"
        )
        last_err = None
        for _ in range(3):
            try:
                s = requests.Session()
                s.headers.update(headers)
                s.get("https://www.nseindia.com", timeout=25)
                r = s.get(url, timeout=25)
                r.raise_for_status()
                j = r.json()
                if j.get("error"):
                    msg = str(j.get("showMessage") or "")
                    if "No record" in msg or "No records" in msg:
                        return empty_df()
                    raise ValueError(msg or "NSE API error")
                data = j.get("data", [])
                if not data:
                    return empty_df()
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["CH_TIMESTAMP"], errors="coerce").dt.date
                df["close"] = pd.to_numeric(df["CH_CLOSING_PRICE"], errors="coerce")
                df["volume"] = pd.to_numeric(df["CH_TOT_TRADED_QTY"], errors="coerce")
                return df[["date", "close", "volume"]].dropna()
            except Exception as e:
                last_err = e
                time.sleep(0.35 + random.random() * 0.4)
        st.warning(f"NSE {d1:%d %b %Y} → {d2:%d %b %Y}: {type(last_err).__name__}: {last_err}")
        return empty_df()

    windows: List[Tuple[dt.date, dt.date]] = []
    if total_days <= 365:
        windows.append((start, end))
    else:
        e = end
        while e >= start:
            s = max(start, e - dt.timedelta(days=364))
            windows.append((s, e))
            e = s - dt.timedelta(days=1)

    parts: List[pd.DataFrame] = []
    for (d1, d2) in windows:
        dfp = fetch_range(d1, d2)
        if not dfp.empty:
            parts.append(dfp)

    out = pd.concat(parts, ignore_index=True) if parts else empty_df()

    if not out.empty:
        def first_of_month(d: dt.date) -> dt.date:
            return dt.date(d.year, d.month, 1)
        expected_ym = {(d.year, d.month) for d in pd.date_range(first_of_month(start), first_of_month(end), freq="MS").date}
        have_ym = set()
        for d_ in pd.to_datetime(out["date"]).dt.to_pydatetime():
            have_ym.add((d_.year, d_.month))
        missing = sorted(expected_ym - have_ym)
        if missing:
            retry_parts: List[pd.DataFrame] = []
            for (yy, mm) in missing:
                d1 = dt.date(yy, mm, 1)
                d2 = (dt.date(yy + (1 if mm == 12 else 0), 1 if mm == 12 else mm + 1, 1) - dt.timedelta(days=1))
                d1 = max(d1, start)
                d2 = min(d2, end)
                if d1 <= d2:
                    retry_parts.append(fetch_range(d1, d2))
            retry_parts = [p for p in retry_parts if not p.empty]
            if retry_parts:
                out = pd.concat([out] + retry_parts, ignore_index=True)

    if out.empty:
        return out.astype({"close": "float64", "volume": "float64"})

    out = (
        out.drop_duplicates(subset=["date"])
           .sort_values("date")
           .reset_index(drop=True)
           .astype({"close": "float64", "volume": "float64"})
    )
    return out

# ---------------- Supabase-backed cache helpers -----------------

def _key_nse(symbol: str, series: str) -> str:
    return f"NSE:{symbol}:{series}".upper()

def _key_bse(scrip: str) -> str:
    return f"BSE:{scrip}"

def get_nse_data_db_cached(
    entity_name: str, symbol: str, series: str, start: dt.date, end: dt.date,
    *, use_cache: bool = True, coverage: str = "daily", tail_only: bool = True
) -> pd.DataFrame:
    k = _key_nse(symbol, series)
    if not symbol or not series:
        return pd.DataFrame(columns=["date", "close", "volume"])

    # Decide what we need to fetch
    fresh_parts: List[pd.DataFrame] = []
    if use_cache:
        if tail_only:
            max_dt = sb_max_date_for_key(k)
            todo = (_tail_windows_monthly(start, end, max_dt)
                    if coverage == "monthly" else
                    _tail_windows_daily(start, end, max_dt))
        else:
            if coverage == "monthly":
                have_months = sb_months_in_range(k, start, end)
                todo = list(month_ranges_missing_months(start, end, have_months))
            else:
                have_days = sb_dates_in_range(k, start, end)
                todo = list(month_ranges_to_fetch(start, end, have_days))
    else:
        todo = [(start, end)]

    for d1, d2 in todo:
        df = get_nse_data(symbol, series, d1, d2)
        if not df.empty:
            fresh_parts.append(df)

    # Upsert any fresh data
    if fresh_parts and use_cache:
        all_new = pd.concat(fresh_parts, ignore_index=True)
        rows = all_new.rename(columns={"date": "dt"})[["dt", "close", "volume"]].copy()
        rows["exchange"] = "NSE"; rows["entity"] = entity_name; rows["k"] = k
        sb_upsert_trades(rows)

    # Always read final frame from DB if cache is on
    return (
        sb_load_range(k, start, end).rename(columns={"dt": "date"})
        if use_cache else
        (pd.concat(fresh_parts, ignore_index=True) if fresh_parts else pd.DataFrame(columns=["date", "close", "volume"]))
    )

def get_bse_data_db_cached(
    entity_name: str, scrip: str, start: dt.date, end: dt.date,
    *, use_cache: bool = True, coverage: str = "daily", tail_only: bool = True
) -> pd.DataFrame:
    k = _key_bse(scrip)
    if not scrip:
        return pd.DataFrame(columns=["date", "close", "volume"])

    fresh_parts: List[pd.DataFrame] = []
    if use_cache:
        if tail_only:
            max_dt = sb_max_date_for_key(k)
            todo = (_tail_windows_monthly(start, end, max_dt)
                    if coverage == "monthly" else
                    _tail_windows_daily(start, end, max_dt))
        else:
            if coverage == "monthly":
                have_months = sb_months_in_range(k, start, end)
                todo = list(month_ranges_missing_months(start, end, have_months))
            else:
                have_days = sb_dates_in_range(k, start, end)
                todo = list(month_ranges_to_fetch(start, end, have_days))
    else:
        todo = [(start, end)]

    for d1, d2 in todo:
        df = get_bse_data(scrip, d1, d2)
        if not df.empty:
            fresh_parts.append(df)

    if fresh_parts and use_cache:
        all_new = pd.concat(fresh_parts, ignore_index=True)
        rows = all_new.rename(columns={"date": "dt"})[["dt", "close", "volume"]].copy()
        rows["exchange"] = "BSE"; rows["entity"] = entity_name; rows["k"] = k
        sb_upsert_trades(rows)

    return (
        sb_load_range(k, start, end).rename(columns={"dt": "date"})
        if use_cache else
        (pd.concat(fresh_parts, ignore_index=True) if fresh_parts else pd.DataFrame(columns=["date", "close", "volume"]))
    )

# -------------------------- one-entity fetch --------------------------

def fetch_single_entity(
    row: pd.Series,
    start: dt.date,
    end: dt.date,
    monthly_mode: bool,
    use_cache_now: bool,
):
    """Return (nse_df, bse_df) for the entity."""
    cov = "monthly" if monthly_mode else "daily"

    sym = row.get("NSE Symbol", "")
    series = row.get("NSE Series", "")
    scrip = row.get("BSE Scrip Code", "")

    nse_df = get_nse_data_db_cached(
        row["Name of Entity"], sym, series, start, end,
        use_cache=use_cache_now, coverage=cov, tail_only=True,
    )
    bse_df = get_bse_data_db_cached(
        row["Name of Entity"], scrip, start, end,
        use_cache=use_cache_now, coverage=cov, tail_only=True,
    )

    if monthly_mode:
        if not nse_df.empty:
            nse_df = to_monthly(nse_df)
        if not bse_df.empty:
            bse_df = to_monthly(bse_df)

    for df in (nse_df, bse_df):
        if not df.empty:
            df["Entity"] = row["Name of Entity"]
            df["Type"] = row["Type of Entity"]

    return nse_df, bse_df

# ------------------------------- render ------------------------------

def render():
    st.header("Trading (NSE & BSE)")

    with st.sidebar:
        st.markdown("### Trading — Controls")
        entities_url = st.text_input(
            "Entities Google Sheet (CSV view)",
            value=ENTITIES_SHEET_CSV,
            help="Must include: Type of Entity, Name of Entity, NSE Symbol, NSE Series, BSE Scrip Code",
            key="entities_csv_url",
        )

        default_start = dt.date(2024, 4, 1)
        default_end = dt.date.today()
        start_date = st.date_input("From", value=default_start, format="DD/MM/YYYY", key="trade_from")
        end_date   = st.date_input("To",   value=default_end,   format="DD/MM/YYYY", key="trade_to")
        start_date, end_date = clamp_dates(start_date, end_date)

        mode = st.radio("Mode", ["Single Entity", "All REITs", "All InvITs"], key="trade_mode")

        # Monthly toggle only for single entity; group views are forced monthly
        if mode == "Single Entity":
            monthly_mode = st.checkbox(
                "Monthly aggregation",
                value=False,
                help="volume=sum, close=last day of month",
                key="trade_monthly",
            )
        else:
            monthly_mode = True
            st.caption("Group view uses monthly aggregation by design.")

        # Keep a simple “use cache” switch (DB first + tail fill). Others removed.
        use_db_cache = st.checkbox(
            "Use cloud cache (Supabase)",
            value=True,
            help="Read from DB first and fetch only the missing tail.",
            key="trade_use_db",
        )

        # Healthcheck (only to show status text)
        sb_ok = sb_healthcheck() if use_db_cache else False
        if use_db_cache:
            st.caption("Supabase: " + ("✅ connected" if sb_ok else "❌ not reachable — falling back to live APIs"))
        st.markdown("---")

        entity_name = None
        ents_all = load_entities(entities_url) if entities_url.strip() else pd.DataFrame()
        if mode == "Single Entity":
            if ents_all.empty:
                st.error("No entities found in the Google Sheet.")
            else:
                ents_ordered = pd.concat(
                    [ents_all[ents_all["Type of Entity"] == "REIT"],
                     ents_all[ents_all["Type of Entity"] == "InvIT"]],
                    ignore_index=True,
                )
                entity_name = st.selectbox("Select Entity", ents_ordered["Name of Entity"].tolist(), index=0)

        st.button("Load / Refresh", key="trade_go")

    if not st.session_state.get("trade_go"):
        st.info("Choose a mode, pick your dates, then click **Load / Refresh**.")
        return

    if ents_all.empty:
        st.error("Could not load entities list. Please check the CSV URL.")
        return

    use_cache_now = use_db_cache and sb_ok

    # -------- Single entity --------
    if mode == "Single Entity":
        if not entity_name:
            st.error("Please select an entity.")
            return
        row = ents_all[ents_all["Name of Entity"] == entity_name].iloc[0]

        nse_df, bse_df = fetch_single_entity(
            row, start_date, end_date, monthly_mode, use_cache_now
        )

        sym, series = row.get("NSE Symbol", ""), row.get("NSE Series", "")
        scrip = row.get("BSE Scrip Code", "")

        col1, col2 = st.columns((1, 1), gap="large")

        with col1:
            if nse_df.empty:
                st.warning("NSE: No data.")
            else:
                st.plotly_chart(
                    line_bar_figure(
                        nse_df,
                        f"NSE • {row['Name of Entity']} • {sym} ({series})" + (" • Monthly" if monthly_mode else ""),
                        height=560 if not monthly_mode else 520,
                        monthly=monthly_mode,
                    ),
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True},
                )
                st.caption(f"Rows: {len(nse_df)} | Range: {nse_df['date'].min()} → {nse_df['date'].max()}")
                st.dataframe(nse_df, use_container_width=True, hide_index=True)

        with col2:
            if bse_df.empty:
                st.warning("BSE: No data.")
            else:
                st.plotly_chart(
                    line_bar_figure(
                        bse_df,
                        f"BSE • {row['Name of Entity']} • {scrip}" + (" • Monthly" if monthly_mode else ""),
                        height=560 if not monthly_mode else 520,
                        monthly=monthly_mode,
                    ),
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True},
                )
                st.caption(f"Rows: {len(bse_df)} | Range: {bse_df['date'].min()} → {bse_df['date'].max()}")
                st.dataframe(bse_df, use_container_width=True, hide_index=True)

    # -------- Group view (monthly enforced) --------
    else:
        group_type = "REIT" if mode == "All REITs" else "InvIT"
        st.subheader(f"Group View • {group_type}s (Monthly)")
        rows = ents_all[ents_all["Type of Entity"].str.upper() == group_type.upper()]
        if rows.empty:
            st.warning(f"No {group_type}s found in the entities sheet.")
            return

        nse_parts, bse_parts = [], []
        for _, r in rows.iterrows():
            nse_df, bse_df = fetch_single_entity(
                r, start_date, end_date, monthly_mode=True, use_cache_now=use_cache_now
            )
            if not nse_df.empty: nse_parts.append(nse_df)
            if not bse_df.empty: bse_parts.append(bse_df)

        nse_all = pd.concat(nse_parts, ignore_index=True) if nse_parts else pd.DataFrame(columns=["date","close","volume","Entity","Type"])
        bse_all = pd.concat(bse_parts, ignore_index=True) if bse_parts else pd.DataFrame(columns=["date","close","volume","Entity","Type"])
        combined_all = pd.concat(
            [nse_all.assign(Exchange="NSE"), bse_all.assign(Exchange="BSE")],
            ignore_index=True
        )

        tabs = st.tabs([
            "NSE (per entity)",
            "BSE (per entity)",
            "Aggregated Volume – NSE",
            "Aggregated Volume – BSE",
            "BSE & NSE (per entity)",
            "Aggregated Volume – BSE & NSE",
        ])

        with tabs[0]:
            if nse_all.empty:
                st.info("NSE: No data.")
            else:
                st.dataframe(nse_all.sort_values(["Entity","date"]), use_container_width=True, hide_index=True)

        with tabs[1]:
            if bse_all.empty:
                st.info("BSE: No data.")
            else:
                st.dataframe(bse_all.sort_values(["Entity","date"]), use_container_width=True, hide_index=True)

        with tabs[2]:
            if nse_all.empty:
                st.info("NSE: No data for aggregate volume.")
            else:
                nse_agg = aggregate_volume_and_turnover(nse_all, monthly=True)
                st.plotly_chart(
                    volume_only_bar(nse_agg[["date","volume"]], f"NSE • Total Volume • All {group_type}s • Monthly", monthly=True),
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True},
                )
                st.dataframe(nse_agg, use_container_width=True, hide_index=True)

        with tabs[3]:
            if bse_all.empty:
                st.info("BSE: No data for aggregate volume.")
            else:
                bse_agg = aggregate_volume_and_turnover(bse_all, monthly=True)
                st.plotly_chart(
                    volume_only_bar(bse_agg[["date","volume"]], f"BSE • Total Volume • All {group_type}s • Monthly", monthly=True),
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True},
                )
                st.dataframe(bse_agg, use_container_width=True, hide_index=True)

        with tabs[4]:
            if combined_all.empty:
                st.info("No data (BSE+NSE) to show for per-entity view.")
            else:
                st.dataframe(
                    combined_all.sort_values(["Entity", "date", "Exchange"]),
                    use_container_width=True,
                    hide_index=True,
                )

        with tabs[5]:
            if combined_all.empty:
                st.info("No data (BSE+NSE) to aggregate.")
            else:
                both_agg = aggregate_volume_and_turnover(combined_all, monthly=True)
                st.plotly_chart(
                    volume_only_bar(
                        both_agg[["date", "volume"]],
                        f"BSE + NSE • Total Volume • All {group_type}s • Monthly",
                        monthly=True,
                    ),
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True},
                )
                st.dataframe(both_agg, use_container_width=True, hide_index=True)
