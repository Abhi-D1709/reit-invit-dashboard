# tabs/trading.py
# Trading (NSE & BSE) — API-only (no DB), with robust NSE NextApi fetch & 1Y windowing

import io
import json
import re
import time
import random
import datetime as dt
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests import HTTPError
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Your common utils should expose the Entities master CSV export URL
# e.g., ENTITIES_SHEET_CSV = "https://docs.google.com/spreadsheets/d/<id>/export?format=csv"
from utils.common import ENTITIES_SHEET_CSV


# ============================== Helpers ==============================
def _clean_str(x):
    return str(x).strip() if pd.notna(x) else ""


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


@st.cache_data(ttl=60 * 30, show_spinner="Loading entity list...")
def load_entities(url: str) -> pd.DataFrame:
    cols = ["Type of Entity", "Name of Entity", "NSE Symbol", "NSE Series", "BSE Scrip Code"]
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load Google Sheet: {e}")
        return pd.DataFrame(columns=cols)

    df = df[[c for c in cols if c in df.columns]].copy()
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
    """
    Monthly per entity:
      volume = sum in month
      close  = last trading day's close in month
      date   = that last trading day
    """
    if df.empty:
        return df
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"]).sort_values("date")
    t["ym"] = t["date"].dt.to_period("M")

    # sum monthly volume
    vol = t.groupby("ym", as_index=False)["volume"].sum()

    # last trading day per month (robust across pandas versions)
    last_idx = t.groupby("ym")["date"].idxmax()
    last_rows = t.loc[last_idx, ["ym", "date", "close"]]

    m = pd.merge(last_rows, vol, on="ym", how="inner")
    m = m.sort_values("date")[["date", "close", "volume"]]
    m["date"] = m["date"].dt.date
    return m


def aggregate_volume_and_turnover(df: pd.DataFrame, monthly: bool) -> pd.DataFrame:
    """
    Aggregate across ALL entities.
      monthly=True:
        volume=sum(volume), turnover=sum(close*volume), avg_daily_turnover=turnover/22, date=month-end
      monthly=False:
        one row per calendar day: volume=sum, turnover=sum(close*volume)
    """
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
        g = t.groupby(t["date"].dt.date, as_index=False).agg(
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
        fig.update_xaxes(
            tickformat="%b %Y",
            dtick="M1",
            ticklabelmode="period",
            tickangle=45,  # diagonal month labels
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
    fig.add_bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.5, marker_line_width=0)
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["close"], name="Close", mode="lines", line=dict(width=2)),
        secondary_y=True,
    )
    # When monthly=True, DO NOT hide weekends (month-ends can be Sat/Sun)
    _base_fig_layout(fig, title, height, hide_weekends=not monthly, monthly=monthly)
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
    return fig


def clamp_dates(start: dt.date, end: dt.date) -> Tuple[dt.date, dt.date]:
    today = dt.date.today()
    if end > today:
        end = today
    if start > end:
        start = end
    return start, end


# ============================== BSE fetch (unchanged URL) ==============================
@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_bse_data(scripcode: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    BSE 'StockReachGraph' endpoint (unchanged).
    """
    scripcode = _normalize_bse_code(scripcode or "")
    if not scripcode:
        return pd.DataFrame(columns=["date", "close", "volume"])
    url = (
        "https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w"
        f"?scripcode={scripcode}&flag=1&fromdate={start.strftime('%Y%m%d')}"
        f"&todate={end.strftime('%Y%m%d')}&seriesid="
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
    }
    try:
        with requests.Session() as s:
            # cookie warm-up
            s.get("https://www.bseindia.com/", headers=headers, timeout=20)
            r = s.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            payload = r.json()
            raw = payload.get("Data", "[]")
            series = json.loads(raw) if isinstance(raw, str) else (raw or [])
            if not isinstance(series, list) or not series:
                return pd.DataFrame(columns=["date", "close", "volume"])
            df = pd.DataFrame(series)
            if not {"dttm", "vale1", "vole"}.issubset(df.columns):
                return pd.DataFrame(columns=["date", "close", "volume"])
            df["date"] = pd.to_datetime(df["dttm"], errors="coerce").dt.date
            df["close"] = pd.to_numeric(df["vale1"], errors="coerce")
            df["volume"] = pd.to_numeric(df["vole"], errors="coerce")
            return df[["date", "close", "volume"]].dropna().sort_values("date")
    except Exception:
        return pd.DataFrame(columns=["date", "close", "volume"])


# ============================== NSE NextApi (with anti-403 hardening) ==============================
@st.cache_resource(show_spinner=False)
def get_nse_session() -> requests.Session:
    """
    Reuse a single warmed-up session so Akamai cookies persist across windows.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/csv, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Origin": "https://www.nseindia.com",
        "Referer": "https://www.nseindia.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    })
    try:
        s.get("https://www.nseindia.com", timeout=20)
    except Exception:
        pass
    return s


def _parse_nse_df_from_jsonlike(data) -> pd.DataFrame:
    """
    Normalize JSON-shaped payload into date, close, volume.
    Accepts dicts with 'data' list or a raw list.
    """
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            df = pd.DataFrame(data["data"])
        else:
            lst = next((v for v in data.values() if isinstance(v, list)), None)
            df = pd.DataFrame(lst) if lst is not None else pd.DataFrame()
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame()

    if df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])

    # Detect columns
    date_cols = [c for c in df.columns if c.upper() in ("CH_TIMESTAMP", "TIMESTAMP", "DATE")]
    close_cols = [c for c in df.columns if c.upper() in ("CH_CLOSING_PRICE", "CLOSE", "CLOSE_PRICE", "CLOSEPRICE")]
    vol_cols = [c for c in df.columns if c.upper() in ("CH_TOT_TRADED_QTY", "VOLUME", "TOTALTRADEDQTY", "TOTAL_TRADED_QUANTITY")]

    if not date_cols:
        return pd.DataFrame(columns=["date", "close", "volume"])
    dcol = date_cols[0]
    ccol = (close_cols[0] if close_cols else next((c for c in df.columns if "close" in c.lower()), None))
    vcol = (vol_cols[0] if vol_cols else next((c for c in df.columns if ("qty" in c.lower() or "volume" in c.lower())), None))
    if not (ccol and vcol):
        return pd.DataFrame(columns=["date", "close", "volume"])

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[dcol], errors="coerce").dt.date
    out["close"] = pd.to_numeric(df[ccol], errors="coerce")
    out["volume"] = pd.to_numeric(df[vcol], errors="coerce")
    return out.dropna()


def _parse_nse_df_from_csv(text: str) -> pd.DataFrame:
    """Parse CSV text returned by NextApi (csv=true)."""
    if not text or not text.strip():
        return pd.DataFrame(columns=["date", "close", "volume"])
    try:
        csv_df = pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame(columns=["date", "close", "volume"])

    cols = {c.lower().strip(): c for c in csv_df.columns}
    date_c = next((cols[k] for k in ("ch_timestamp", "timestamp", "date", "trade_date", "tradedate") if k in cols), None)
    close_c = next((cols[k] for k in ("ch_closing_price", "close", "close_price", "closeprice") if k in cols), None)
    vol_c = next((cols[k] for k in ("ch_tot_traded_qty", "total_traded_quantity", "volume", "tottrdqty", "totaltradedqty") if k in cols), None)

    # fuzzy fallback
    if not date_c:
        date_c = next((c for c in csv_df.columns if "date" in c.lower() or "timestamp" in c.lower()), None)
    if not close_c:
        close_c = next((c for c in csv_df.columns if "close" in c.lower()), None)
    if not vol_c:
        vol_c = next((c for c in csv_df.columns if "qty" in c.lower() or "volume" in c.lower()), None)

    if not (date_c and close_c and vol_c):
        return pd.DataFrame(columns=["date", "close", "volume"])

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(csv_df[date_c], errors="coerce").dt.date
    out["close"] = pd.to_numeric(csv_df[close_c], errors="coerce")
    out["volume"] = pd.to_numeric(csv_df[vol_c], errors="coerce")
    return out.dropna()


@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_nse_data(symbol: str, series: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    NSE via NextApi:
      https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?functionName=getHistoricalTradeData
        &symbol=EMBASSY&series=RR&fromDate=10-12-2024&toDate=10-12-2025&csv=true

    Rules:
      * If range > 1 year, fetch in ≤1-year windows (walk backwards from end), reusing a warmed session.
      * On 403 with csv=true, retry that window with csv=false (JSON).
    """
    if not symbol or not series or start > end:
        return pd.DataFrame(columns=["date", "close", "volume"])

    base = "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"

    # Build ≤1-year windows backwards from end
    windows: List[Tuple[dt.date, dt.date]] = []
    e = end
    while e >= start:
        s_win = max(start, e - dt.timedelta(days=364))
        windows.append((s_win, e))
        e = s_win - dt.timedelta(days=1)

    sess = get_nse_session()
    quote_url = f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"

    # Per-symbol warm-up
    try:
        sess.get(quote_url, timeout=20)
    except Exception:
        pass

    def fetch_window(d1: dt.date, d2: dt.date) -> pd.DataFrame:
        last_err = None
        for _ in range(3):
            params_csv = {
                "functionName": "getHistoricalTradeData",
                "symbol": symbol,
                "series": series,
                "fromDate": d1.strftime("%d-%m-%Y"),
                "toDate": d2.strftime("%d-%m-%Y"),
                "csv": "true",
            }
            try:
                r = sess.get(base, params=params_csv, timeout=25, headers={"Referer": quote_url})
                r.raise_for_status()
                ctype = (r.headers.get("content-type") or "").lower()

                # JSON-ish path (csv=true might still return JSON)
                if "application/json" in ctype or r.text.lstrip("\ufeff").strip().startswith("{"):
                    try:
                        j = r.json()
                    except json.JSONDecodeError:
                        j = json.loads(r.text.lstrip("\ufeff").strip())
                    dfj = _parse_nse_df_from_jsonlike(j)
                    if not dfj.empty:
                        return dfj

                # CSV/text path
                dfc = _parse_nse_df_from_csv(r.text)
                if not dfc.empty:
                    return dfc

                # If neither parsed, raise to trigger fallback/retry
                raise ValueError("Unexpected NSE response format")

            except HTTPError as e:
                last_err = e
                if getattr(e.response, "status_code", None) == 403:
                    # Refresh cookies + retry with csv=false once
                    try:
                        sess.get("https://www.nseindia.com", timeout=20)
                        sess.get(quote_url, timeout=20)
                    except Exception:
                        pass
                    try:
                        params_json = {
                            "functionName": "getHistoricalTradeData",
                            "symbol": symbol,
                            "series": series,
                            "fromDate": d1.strftime("%d-%m-%Y"),
                            "toDate": d2.strftime("%d-%m-%Y"),
                            "csv": "false",
                        }
                        r2 = sess.get(base, params=params_json, timeout=25, headers={"Referer": quote_url})
                        r2.raise_for_status()
                        j = r2.json()
                        dfj = _parse_nse_df_from_jsonlike(j)
                        if not dfj.empty:
                            return dfj
                    except Exception as e2:
                        last_err = e2
                time.sleep(0.6 + random.random() * 0.8)
            except Exception as e:
                last_err = e
                time.sleep(0.6 + random.random() * 0.8)

        st.warning(f"NSE {symbol} {d1:%d-%b-%Y}→{d2:%d-%b-%Y}: {type(last_err).__name__}: {last_err}")
        return pd.DataFrame(columns=["date", "close", "volume"])

    parts = []
    for d1, d2 in windows:
        parts.append(fetch_window(d1, d2))
        # small pacing between windows to avoid rate-limit bursts
        time.sleep(0.4 + random.random() * 0.6)

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["date", "close", "volume"])

    out = (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=["date"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["close", "volume"])
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


# ============================== Per-entity fetch ==============================
def fetch_single_entity(
    row: pd.Series, start: dt.date, end: dt.date, monthly_mode: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (nse_df, bse_df) for a given row from entities table (API-only).
    """
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


# ============================== UI Components ==============================
def render_sidebar(entities_df: pd.DataFrame):
    st.markdown("### Trading — Controls")
    start, end = clamp_dates(
        st.date_input("From", value=dt.date(2024, 4, 1), format="DD/MM/YYYY", key="trade_from"),
        st.date_input("To", value=dt.date.today(), format="DD/MM/YYYY", key="trade_to"),
    )
    mode = st.radio("Mode", ["Single Entity", "All REITs", "All InvITs"], key="trade_mode", horizontal=True)
    monthly_mode = st.checkbox(
        "Monthly aggregation",
        value=(mode != "Single Entity"),
        help="volume=sum, close=last day of month",
        key="trade_monthly",
        disabled=(mode != "Single Entity"),
    )

    entity_name = None
    if mode == "Single Entity" and not entities_df.empty:
        # Keep displayed order as in sheet
        entity_name = st.selectbox("Select Entity", entities_df["Name of Entity"].tolist(), index=0)
    return mode, start, end, (monthly_mode or mode != "Single Entity"), entity_name


def render_single_entity_view(row, start_date, end_date, monthly_mode):
    st.subheader(f"Entity View: {row['Name of Entity']}")
    nse_df, bse_df = fetch_single_entity(row, start_date, end_date, monthly_mode)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        title = f"NSE • {row.get('NSE Symbol', '')}" + (" (Monthly)" if monthly_mode else "")
        if nse_df.empty:
            st.warning("NSE: No data for this period.")
        else:
            st.plotly_chart(line_bar_figure(nse_df, title, monthly=monthly_mode), width="stretch",
                            config={"displayModeBar": True, "scrollZoom": True})
            st.dataframe(nse_df, width="stretch", hide_index=True)
            st.download_button(
                "Download NSE CSV" + (" (Monthly)" if monthly_mode else ""),
                nse_df.to_csv(index=False).encode("utf-8"),
                file_name=f"NSE_{row.get('NSE Symbol','')}_{row.get('NSE Series','')}_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                mime="text/csv",
            )
    with c2:
        title = f"BSE • {row.get('BSE Scrip Code', '')}" + (" (Monthly)" if monthly_mode else "")
        if bse_df.empty:
            st.warning("BSE: No data for this period.")
        else:
            st.plotly_chart(line_bar_figure(bse_df, title, monthly=monthly_mode), width="stretch",
                            config={"displayModeBar": True, "scrollZoom": True})
            st.dataframe(bse_df, width="stretch", hide_index=True)
            st.download_button(
                "Download BSE CSV" + (" (Monthly)" if monthly_mode else ""),
                bse_df.to_csv(index=False).encode("utf-8"),
                file_name=f"BSE_{row.get('BSE Scrip Code','')}_{start_date}_{end_date}{'_monthly' if monthly_mode else ''}.csv",
                mime="text/csv",
            )


# ============================== Main Render ==============================
def render():
    st.header("Trading (NSE & BSE)")

    entities_all = load_entities(ENTITIES_SHEET_CSV)

    with st.sidebar:
        mode, start_date, end_date, monthly_mode, entity_name = render_sidebar(entities_all)

    if entities_all.empty:
        st.error("Could not load entities list. Please check the Entities Sheet URL.")
        return

    if mode == "Single Entity":
        if not entity_name:
            st.info("Select an entity from the sidebar to begin.")
            return
        selected_row = entities_all[entities_all["Name of Entity"] == entity_name].iloc[0]
        render_single_entity_view(selected_row, start_date, end_date, monthly_mode)
        return

    # --------- Group View ---------
    group_type = "REIT" if mode == "All REITs" else "InvIT"
    st.subheader(f"Group View • All {group_type}s")
    rows_to_fetch = entities_all[entities_all["Type of Entity"].str.upper() == group_type.upper()]

    if rows_to_fetch.empty:
        st.warning(f"No {group_type}s found in the entities sheet.")
        return

    # Always fetch DAILY first; aggregate to monthly for charts in group view
    daily_nse_parts, daily_bse_parts = [], []
    with st.status(f"Fetching daily data for {len(rows_to_fetch)} {group_type}s...", expanded=True) as status:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futs = {executor.submit(fetch_single_entity, r, start_date, end_date, False): r for _, r in rows_to_fetch.iterrows()}
            for fut in as_completed(futs):
                r = futs[fut]
                name = r["Name of Entity"]
                try:
                    nse_df, bse_df = fut.result()
                    if not nse_df.empty:
                        daily_nse_parts.append(nse_df)
                    if not bse_df.empty:
                        daily_bse_parts.append(bse_df)
                    status.write(f"✅ {name}")
                except Exception as e:
                    status.write(f"⌐ {name}: {e}")
        status.update(label="All data loaded!", state="complete", expanded=False)

    daily_nse_all = pd.concat(daily_nse_parts, ignore_index=True) if daily_nse_parts else pd.DataFrame()
    daily_bse_all = pd.concat(daily_bse_parts, ignore_index=True) if daily_bse_parts else pd.DataFrame()
    combined_all_daily = pd.concat(
        [
            daily_nse_all.assign(Exchange="NSE") if not daily_nse_all.empty else pd.DataFrame(),
            daily_bse_all.assign(Exchange="BSE") if not daily_bse_all.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )

    tabs = st.tabs([
        "NSE (per entity)",
        "BSE (per entity)",
        "BSE & NSE (per entity)",
        "Aggregated Volume – NSE",
        "Aggregated Volume – BSE",
        "Aggregated Volume – BSE & NSE",
    ])

    # NSE per entity (daily)
    with tabs[0]:
        st.subheader("NSE • Daily (per entity)")
        if daily_nse_all.empty:
            st.info("NSE: No data.")
        else:
            st.dataframe(daily_nse_all.sort_values(["Entity", "date"]), width="stretch", hide_index=True)
            st.download_button(
                "Download NSE (Per Entity, Daily)",
                daily_nse_all.to_csv(index=False).encode("utf-8"),
                file_name=f"NSE_{group_type}_PER_ENTITY_{start_date}_{end_date}.csv",
                mime="text/csv",
            )

    # BSE per entity (daily)
    with tabs[1]:
        st.subheader("BSE • Daily (per entity)")
        if daily_bse_all.empty:
            st.info("BSE: No data.")
        else:
            st.dataframe(daily_bse_all.sort_values(["Entity", "date"]), width="stretch", hide_index=True)
            st.download_button(
                "Download BSE (Per Entity, Daily)",
                daily_bse_all.to_csv(index=False).encode("utf-8"),
                file_name=f"BSE_{group_type}_PER_ENTITY_{start_date}_{end_date}.csv",
                mime="text/csv",
            )

    # BSE & NSE (per entity) combined (daily)
    with tabs[2]:
        st.subheader("BSE & NSE • Daily (per entity)")
        if combined_all_daily.empty:
            st.info("No data.")
        else:
            st.dataframe(
                combined_all_daily.sort_values(["Entity", "date", "Exchange"]),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download BSE+NSE (Per Entity, Daily)",
                combined_all_daily.to_csv(index=False).encode("utf-8"),
                file_name=f"BSE_NSE_{group_type}_PER_ENTITY_{start_date}_{end_date}.csv",
                mime="text/csv",
            )

    # Aggregated Volume – NSE (monthly)
    with tabs[3]:
        st.subheader("Aggregated Volume – NSE (Monthly)")
        if daily_nse_all.empty:
            st.info("NSE: No data for aggregate volume.")
        else:
            nse_agg = aggregate_volume_and_turnover(daily_nse_all, monthly=True)
            st.plotly_chart(
                volume_only_bar(nse_agg[["date", "volume"]], f"NSE • Total Volume • All {group_type}s • Monthly", monthly=True),
                width="stretch",
                config={"displayModeBar": True, "scrollZoom": True},
            )
            st.dataframe(nse_agg, width="stretch", hide_index=True)
            st.download_button(
                "Download NSE Aggregate (Monthly)",
                nse_agg.to_csv(index=False).encode("utf-8"),
                file_name=f"NSE_{group_type}_AGG_{start_date}_{end_date}_monthly.csv",
                mime="text/csv",
            )

    # Aggregated Volume – BSE (monthly)
    with tabs[4]:
        st.subheader("Aggregated Volume – BSE (Monthly)")
        if daily_bse_all.empty:
            st.info("BSE: No data for aggregate volume.")
        else:
            bse_agg = aggregate_volume_and_turnover(daily_bse_all, monthly=True)
            st.plotly_chart(
                volume_only_bar(bse_agg[["date", "volume"]], f"BSE • Total Volume • All {group_type}s • Monthly", monthly=True),
                width="stretch",
                config={"displayModeBar": True, "scrollZoom": True},
            )
            st.dataframe(bse_agg, width="stretch", hide_index=True)
            st.download_button(
                "Download BSE Aggregate (Monthly)",
                bse_agg.to_csv(index=False).encode("utf-8"),
                file_name=f"BSE_{group_type}_AGG_{start_date}_{end_date}_monthly.csv",
                mime="text/csv",
            )

    # Aggregated Volume – BSE & NSE (monthly)
    with tabs[5]:
        st.subheader("Aggregated Volume – BSE & NSE (Monthly)")
        if combined_all_daily.empty:
            st.info("No data for aggregate volume.")
        else:
            both_agg = aggregate_volume_and_turnover(combined_all_daily, monthly=True)
            st.plotly_chart(
                volume_only_bar(both_agg[["date", "volume"]], f"BSE+NSE • Total Volume • All {group_type}s • Monthly", monthly=True),
                width="stretch",
                config={"displayModeBar": True, "scrollZoom": True},
            )
            st.dataframe(both_agg, width="stretch", hide_index=True)
            st.download_button(
                "Download BSE+NSE Aggregate (Monthly)",
                both_agg.to_csv(index=False).encode("utf-8"),
                file_name=f"BSE_NSE_{group_type}_AGG_{start_date}_{end_date}_monthly.csv",
                mime="text/csv",
            )
