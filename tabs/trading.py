# tabs/trading.py
import json
import re
import datetime as dt
from typing import Optional, Tuple, List, Tuple as Tup

import pandas as pd
import requests
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, random
from decimal import Decimal, InvalidOperation

# Pull the public Google Sheet used for entity master from utils.common
from utils.common import ENTITIES_SHEET_CSV  # sheet id & csv url are defined centrally


# ============================== Helpers (clean/normalize) ==============================
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


# ============================== Entities loader (Google Sheet) ==============================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_entities() -> pd.DataFrame:
    """
    Load the REIT/InvIT mapping from Google Sheets (CSV export).
    NO FALLBACK — if loading fails, return empty and the UI shows an error.
    Expected columns:
      - Type of Entity | Name of Entity | NSE Symbol | NSE Series | BSE Scrip Code
    """
    cols = ["Type of Entity", "Name of Entity", "NSE Symbol", "NSE Series", "BSE Scrip Code"]
    try:
        df = pd.read_csv(ENTITIES_SHEET_CSV)
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


# ============================== Aggregation & Charts ==============================
def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregation per entity: volume=sum, close=last trading day close, date=that last day."""
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
    """
    Aggregate across ALL entities (group view).
    - If monthly=True: one row per month,
        volume = sum(volume)
        turnover = sum(close * volume)
        avg_daily_turnover = turnover / 22
        date = month-end
    - If monthly=False: one row per calendar day,
        volume = sum(volume)
        turnover = sum(close * volume)
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "volume"])

    t = df.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])  # guard
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
        out = (
            g[["date", "volume", "turnover", "avg_daily_turnover"]]
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        # IMPORTANT: name the grouping key explicitly to avoid unnamed-column issues
        t["day"] = t["date"].dt.date
        g = t.groupby("day", as_index=False).agg(
            volume=("volume", "sum"),
            turnover=("turnover", "sum"),
        )
        g = g.rename(columns={"day": "date"})
        out = (
            g[["date", "volume", "turnover"]]
            .sort_values("date")
            .reset_index(drop=True)
        )

    # Normalize numeric dtypes
    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def _apply_xaxis(fig: go.Figure, *, hide_weekends: bool, monthly: bool):
    if hide_weekends:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    if monthly:
        fig.update_xaxes(
            tickformat="%b %Y",
            dtick="M1",
            ticklabelmode="period",
            tickangle=0,
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
    # IMPORTANT: when monthly=True, DO NOT hide weekends (month-ends can be Sat/Sun)
    _base_fig_layout(fig, title, height, hide_weekends=not monthly, monthly=monthly)

    if monthly:
        tickvals = pd.to_datetime(df["date"])
        ticktext = [d.strftime("%b %Y") for d in tickvals]
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45,  # diagonal month labels
        )
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
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45,
        )
    return fig

def clamp_dates(start: dt.date, end: dt.date) -> Tuple[dt.date, dt.date]:
    today = dt.date.today()
    if end > today:
        end = today
    if start > end:
        start = end
    return start, end


# ============================== BSE fetch ==============================
@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_bse_data(scripcode: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Robust BSE fetch with normalization, cookie warm-up, retries, and safe JSON parsing."""
    scripcode = _normalize_bse_code(scripcode or "")
    if not scripcode:
        return pd.DataFrame(columns=["date", "close", "volume"]).astype(
            {"date": "datetime64[ns]", "close": "float64", "volume": "float64"}
        )

    def build_url(flag_val: int) -> str:
        return (
            "https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w"
            f"?scripcode={scripcode}"
            f"&flag={flag_val}"
            f"&fromdate={start.strftime('%Y%m%d')}"
            f"&todate={end.strftime('%Y%m%d')}"
            "&seriesid="
        )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
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
        # cookie warm-up
        try:
            sess.get("https://www.bseindia.com/", timeout=25)
        except Exception:
            pass
        return sess.get(build_url(flag_val), timeout=25)

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
            r = try_fetch(flag_val=1)
            r.raise_for_status()
            j = parse_json_safely(r)
            df = to_df(j)
            if not df.empty:
                return df
            # fallback flag=0 if flag=1 empty
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


# ============================== NSE fetch (hardened for Cloud) ==============================
@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_nse_data(symbol: str, series: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    NSE historical fetch with strict ≤1-year windows:
      • If (end - start) ≤ 365 days: single call.
      • If > 1 year: build ≤360-day windows backwards: [end-359→end], [end-719→end-360], ...
      • If any window returns 403 or 'not more than 1 Year', recursively split that window.
      • One-time monthly gap refill at the end (some months randomly come back empty on Cloud).
    Returns df[date, close, volume] sorted by date.
    """
    if not symbol or not series:
        return pd.DataFrame(columns=["date", "close", "volume"])

    if start > end:
        start, end = end, end

    # Browser-like headers for Streamlit Cloud
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "sec-ch-ua": '"Chromium";v="124", "Not:A-Brand";v="99"',
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
    }
    series_json = json.dumps([series])

    def empty_df() -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "close", "volume"])

    # One session reused across all calls
    sess = requests.Session()
    sess.headers.update(headers)
    try:
        sess.get("https://www.nseindia.com", timeout=25)  # cookie warm-up
    except Exception:
        pass

    def _fetch_range(d1: dt.date, d2: dt.date, depth: int = 0) -> pd.DataFrame:
        """Fetch a single range; on 403/1-year error, split recursively."""
        url = (
            "https://www.nseindia.com/api/historicalOR/cm/equity"
            f"?symbol={symbol}&series={series_json}&from={d1.strftime('%d-%m-%Y')}&to={d2.strftime('%d-%m-%Y')}"
        )

        last_err = None
        for _ in range(5):
            try:
                r = sess.get(url, timeout=25)
                # Explicitly treat 403 as split-worthy (Cloud throttling)
                if r.status_code == 403:
                    raise requests.HTTPError("403 Forbidden", response=r)

                r.raise_for_status()
                j = r.json()

                # NSE "not more than 1 Year" guard
                if j.get("error"):
                    msg = str(j.get("showMessage") or "")
                    if "not more than 1 Year" in msg:
                        # Force split below
                        raise ValueError("Range exceeds 1 year (server)")

                    if "No record" in msg or "No records" in msg:
                        return empty_df()

                    raise ValueError(msg or "NSE API error")

                data = j.get("data", [])
                if not isinstance(data, list) or not data:
                    # Often transient empty payloads—retry
                    raise ValueError("Empty data list")

                df = pd.DataFrame(data)
                needed = {"CH_TIMESTAMP", "CH_CLOSING_PRICE", "CH_TOT_TRADED_QTY"}
                if not needed.issubset(df.columns):
                    raise ValueError("Unexpected schema from NSE")

                df["date"] = pd.to_datetime(df["CH_TIMESTAMP"], errors="coerce").dt.date
                df["close"] = pd.to_numeric(df["CH_CLOSING_PRICE"], errors="coerce")
                df["volume"] = pd.to_numeric(df["CH_TOT_TRADED_QTY"], errors="coerce")
                return df[["date", "close", "volume"]].dropna()

            except Exception as e:
                last_err = e
                # On failure, try re-warming cookies and back off
                try:
                    sess.get("https://www.nseindia.com", timeout=20)
                except Exception:
                    pass
                time.sleep(0.7 + random.random() * 0.8)

        # If we failed all retries: split the window if it is > 45 days OR if server hinted 1-year limit or 403
        span = (d2 - d1).days
        if span > 45 and isinstance(last_err, (requests.HTTPError, ValueError)):
            # Split window in half and try both sides
            mid = d1 + dt.timedelta(days=span // 2)
            left = _fetch_range(d1, mid, depth + 1)
            right = _fetch_range(mid + dt.timedelta(days=1), d2, depth + 1)
            if not left.empty or not right.empty:
                return pd.concat([left, right], ignore_index=True)
        # Give up on this window
        st.warning(f"NSE {d1:%d %b %Y} → {d2:%d %b %Y}: {type(last_err).__name__}: {last_err}")
        return empty_df()

    # Build windows: ≤360 days when period > 1 year, else just one window
    total_days = (end - start).days
    if total_days <= 365:
        windows = [(start, end)]
    else:
        windows: List[Tuple[dt.date, dt.date]] = []
        e = end
        while e >= start:
            s = max(start, e - dt.timedelta(days=360))
            windows.append((s, e))
            e = s - dt.timedelta(days=1)

    # Sequential fetch to avoid rate limits
    parts: List[pd.DataFrame] = []
    for (d1, d2) in windows:
        dfp = _fetch_range(d1, d2)
        if not dfp.empty:
            parts.append(dfp)

    out = pd.concat(parts, ignore_index=True) if parts else empty_df()

    # Monthly gap refill (helps when a month intermittently returns empty on Cloud)
    if not out.empty:
        def first_of_month(d: dt.date) -> dt.date:
            return dt.date(d.year, d.month, 1)

        exp_ym = {(d.year, d.month) for d in pd.date_range(first_of_month(start), first_of_month(end), freq="MS").date}
        have_ym = {(d.year, d.month) for d in pd.to_datetime(out["date"]).dt.date}
        missing = sorted(exp_ym - have_ym)
        if missing:
            retry_parts: List[pd.DataFrame] = []
            for (yy, mm) in missing:
                d1 = dt.date(yy, mm, 1)
                d2 = (dt.date(yy + (1 if mm == 12 else 0), 1 if mm == 12 else mm + 1, 1) - dt.timedelta(days=1))
                d1 = max(d1, start)
                d2 = min(d2, end)
                if d1 <= d2:
                    retry_parts.append(_fetch_range(d1, d2))
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

# ============================== Multi-entity helpers ==============================
def fetch_single_entity(
    row: pd.Series,
    start: dt.date,
    end: dt.date,
    monthly_mode: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (nse_df, bse_df) for a given row from entities table."""
    nse_df = pd.DataFrame(columns=["date", "close", "volume"])
    bse_df = pd.DataFrame(columns=["date", "close", "volume"])

    sym = row.get("NSE Symbol", "")
    series = row.get("NSE Series", "")
    scrip = row.get("BSE Scrip Code", "")

    if sym and series:
        try:
            nse_df = get_nse_data(sym, series, start, end)
            if monthly_mode and not nse_df.empty:
                nse_df = to_monthly(nse_df)
        except Exception as e:
            st.warning(f"NSE fetch failed for {row['Name of Entity']}: {e}")

    if scrip:
        try:
            bse_df = get_bse_data(scrip, start, end)
            if monthly_mode and not bse_df.empty:
                bse_df = to_monthly(bse_df)
        except Exception as e:
            st.warning(f"BSE fetch failed for {row['Name of Entity']}: {e}")

    for df in (nse_df, bse_df):
        if not df.empty:
            df["Entity"] = row["Name of Entity"]
            df["Type"] = row["Type of Entity"]
    return nse_df, bse_df

def fetch_group(
    df_entities: pd.DataFrame,
    entity_type: str,  # "REIT" or "InvIT"
    start: dt.date,
    end: dt.date,
    monthly_mode: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch NSE and BSE for all entities of a given type (parallel per entity)."""
    rows = df_entities[df_entities["Type of Entity"].str.upper() == entity_type.upper()]
    if rows.empty:
        return (pd.DataFrame(columns=["date","close","volume","Entity","Type"]),
                pd.DataFrame(columns=["date","close","volume","Entity","Type"]))

    nse_parts, bse_parts = [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fetch_single_entity, row, start, end, monthly_mode): row["Name of Entity"]
                   for _, row in rows.iterrows()}
        for fut in as_completed(futures):
            nse_df, bse_df = fut.result()
            if not nse_df.empty:
                nse_parts.append(nse_df)
            if not bse_df.empty:
                bse_parts.append(bse_df)

    nse_all = pd.concat(nse_parts, ignore_index=True) if nse_parts else pd.DataFrame(columns=["date","close","volume","Entity","Type"])
    bse_all = pd.concat(bse_parts, ignore_index=True) if bse_parts else pd.DataFrame(columns=["date","close","volume","Entity","Type"])
    return nse_all, bse_all


# ============================== Page renderer ==============================
def render():
    """Entry-point called by pages/4_Trading.py."""
    st.markdown("### REIT / InvIT • NSE & BSE Price–Volume")

    # Sidebar controls
    with st.sidebar:
        st.markdown("#### Date range")
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

        with st.expander("(Optional) Inspect Entity Master"):
            st.caption("Loaded directly from Google Sheets (no fallback).")
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

    # ============================== Main ==============================
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

        matches = df_master[df_master["Name of Entity"] == entity_name]
        if matches.empty:
            st.error("Selected entity not found in the Google Sheet.")
            st.stop()
        row = matches.iloc[0]

        col1, col2 = st.columns((1, 1), gap="large")

        # --------------- NSE ---------------
        with st.spinner(f"Fetching NSE: {row['Name of Entity']}"):
            sym, series = row.get("NSE Symbol", ""), row.get("NSE Series", "")
            try:
                nse_df = get_nse_data(sym, series, start_date, end_date) if (sym and series) else pd.DataFrame(columns=["date","close","volume"])
                if monthly_mode and not nse_df.empty:
                    nse_df = to_monthly(nse_df)
                if nse_df.empty:
                    st.warning("NSE: No data (symbol/series may be missing or no trading data).")
                else:
                    title = f"NSE • {row['Name of Entity']} • {sym} ({series})" + (" • Monthly" if monthly_mode else "")
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
                            file_name=f"NSE_{sym}_{series}_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.exception(e)

        # --------------- BSE ---------------
        with st.spinner(f"Fetching BSE: {row['Name of Entity']}"):
            scrip = row.get("BSE Scrip Code", "")
            try:
                bse_df = get_bse_data(scrip, start_date, end_date) if scrip else pd.DataFrame(columns=["date","close","volume"])
                if monthly_mode and not bse_df.empty:
                    bse_df = to_monthly(bse_df)
                if bse_df.empty:
                    st.warning("BSE: No data (scrip code may be missing or no trading data).")
                else:
                    title = f"BSE • {row['Name of Entity']} • {scrip}" + (" • Monthly" if monthly_mode else "")
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
                            file_name=f"BSE_{scrip}_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.exception(e)

    else:
        group_type = "REIT" if mode == "All REITs" else "InvIT"
        st.subheader(f"Group View • {group_type}s")

        with st.spinner(f"Fetching all {group_type}s... (NSE & BSE)"):
            nse_all, bse_all = fetch_group(df_master, group_type, start_date, end_date, monthly_mode)

        if nse_all.empty and bse_all.empty:
            st.warning("No data returned for the chosen range/group (check symbols/series/scrip codes).")
            return

        # Build combined view (keeps an Exchange column)
        combined_all = pd.concat(
            [nse_all.assign(Exchange="NSE"), bse_all.assign(Exchange="BSE")],
            ignore_index=True,
        )

        tabs = st.tabs([
            "NSE (per entity)",
            "BSE (per entity)",
            "Aggregated Volume – NSE",
            "Aggregated Volume – BSE",
            "BSE & NSE (per entity)",          # Combined per-entity
            "Aggregated Volume – BSE & NSE",   # Combined aggregate
        ])

        # ---- NSE per entity
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

        # ---- BSE per entity
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

        # ---- Aggregated Volume – NSE
        with tabs[2]:
            if nse_all.empty:
                st.info("NSE: No data for aggregate volume.")
            else:
                nse_agg = aggregate_volume_and_turnover(nse_all, monthly_mode)
                st.plotly_chart(
                    volume_only_bar(
                        nse_agg[["date","volume"]],
                        f"NSE • Total Volume • All {group_type}s" + (" • Monthly" if monthly_mode else ""),
                        monthly=monthly_mode
                    ),
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

        # ---- Aggregated Volume – BSE
        with tabs[3]:
            if bse_all.empty:
                st.info("BSE: No data for aggregate volume.")
            else:
                bse_agg = aggregate_volume_and_turnover(bse_all, monthly_mode)
                st.plotly_chart(
                    volume_only_bar(
                        bse_agg[["date","volume"]],
                        f"BSE • Total Volume • All {group_type}s" + (" • Monthly" if monthly_mode else ""),
                        monthly=monthly_mode
                    ),
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

        # ---- BSE & NSE (per entity)  [Combined]
        with tabs[4]:
            if combined_all.empty:
                st.info("No data (BSE+NSE) to show for per-entity view.")
            else:
                st.dataframe(
                    combined_all.sort_values(["Entity", "date", "Exchange"]),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download BSE & NSE (Per Entity)" + (" (Monthly)" if monthly_mode else ""),
                    combined_all.to_csv(index=False).encode("utf-8"),
                    file_name=f"BSE_NSE_{group_type}_PER_ENTITY_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                    mime="text/csv",
                )

        # ---- Aggregated Volume – BSE & NSE  [Combined]
        with tabs[5]:
            if combined_all.empty:
                st.info("No data (BSE+NSE) to aggregate.")
            else:
                both_agg = aggregate_volume_and_turnover(combined_all, monthly_mode)
                st.plotly_chart(
                    volume_only_bar(
                        both_agg[["date", "volume"]],
                        f"BSE + NSE • Total Volume • All {group_type}s" + (" • Monthly" if monthly_mode else ""),
                        monthly=monthly_mode,
                    ),
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True},
                )
                st.dataframe(both_agg, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download BSE + NSE Aggregate" + (" (Monthly + Avg Daily Turnover)" if monthly_mode else ""),
                    both_agg.to_csv(index=False).encode("utf-8"),
                    file_name=f"BSE_NSE_{group_type}_AGG_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                    mime="text/csv",
                )
