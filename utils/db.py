# utils/db.py
import os, json, datetime as dt
from typing import Iterable, Tuple, Set, Optional, Dict, Any

import pandas as pd
import requests
import streamlit as st

# -------------------- credentials --------------------
def _read_sb_creds() -> Tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    try:
        s = st.secrets  # may raise if secrets.toml missing
        url = s.get("SUPABASE_URL", url)
        key = s.get("SUPABASE_KEY", key)
    except Exception:
        pass
    return url, key

SUPABASE_URL, SUPABASE_KEY = _read_sb_creds()
SB_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)

HEADERS_JSON = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# -------------------- tiny logger --------------------
def _log(msg: str) -> None:
    if st.session_state.get("trade_show_sb_logs"):
        st.session_state.setdefault("_sb_logs", []).append(str(msg))

def show_log_widget() -> None:
    if st.session_state.get("_sb_logs"):
        with st.expander("Supabase log", expanded=False):
            for line in st.session_state["_sb_logs"]:
                st.code(line)

# -------------------- healthcheck --------------------
def sb_healthcheck() -> bool:
    """Cheap read to confirm REST is reachable."""
    if not SB_ENABLED:
        return False
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades?select=k&limit=1",
            headers={**HEADERS_JSON, "Prefer": "count=exact"},
            timeout=20,
        )
        _log(f"GET /trades (healthcheck) -> {r.status_code} {r.headers.get('content-range','')}")
        return r.status_code in (200, 206)
    except Exception as e:
        _log(f"HC error: {e}")
        return False

# -------------------- range helpers --------------------
def sb_dates_in_range(k: str, start: dt.date, end: dt.date) -> Set[dt.date]:
    """All trading dates present for key k in [start, end]."""
    if not SB_ENABLED:
        return set()
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades?select=dt&k=eq.{k}&dt=gte.{start}&dt=lte.{end}",
            headers=HEADERS_JSON,
            timeout=40,
        )
        r.raise_for_status()
        rows = r.json() or []
        out = {pd.to_datetime(x.get("dt"), errors="coerce").date() for x in rows if x.get("dt")}
        _log(f"GET dates {k} [{start}..{end}] -> {len(out)} dates")
        return out
    except Exception as e:
        _log(f"GET dates error {k}: {e}")
        return set()

def sb_months_in_range(k: str, start: dt.date, end: dt.date) -> Set[Tuple[int, int]]:
    """Set of (year, month) that have at least one row for key k in [start, end]."""
    if not SB_ENABLED:
        return set()
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades?select=dt&k=eq.{k}&dt=gte.{start}&dt=lte.{end}",
            headers=HEADERS_JSON,
            timeout=40,
        )
        r.raise_for_status()
        rows = r.json() or []
        if not rows:
            return set()
        dts = pd.to_datetime([x.get("dt") for x in rows], errors="coerce")
        ym = {(d.year, d.month) for d in dts.dropna().to_pydatetime()}
        _log(f"GET months {k} -> {len(ym)} months")
        return ym
    except Exception as e:
        _log(f"GET months error {k}: {e}")
        return set()

def month_ranges_to_fetch(start: dt.date, end: dt.date, have_dates: Set[dt.date]) -> Iterable[Tuple[dt.date, dt.date]]:
    """Yield (d1, d2) month windows where at least one **day** is missing."""
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + 1, 1, 1) if cur.month == 12 else dt.date(cur.year, cur.month + 1, 1)
        d1 = max(cur, start)
        d2 = min(nxt - dt.timedelta(days=1), end)
        days = {d1 + dt.timedelta(days=i) for i in range((d2 - d1).days + 1)}
        if not days.issubset(have_dates):
            yield d1, d2
        cur = nxt

def month_ranges_missing_months(start: dt.date, end: dt.date, have_months: Set[Tuple[int,int]]) -> Iterable[Tuple[dt.date, dt.date]]:
    """Yield (first_day,last_day) for months that have **no** row at all."""
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + 1, 1, 1) if cur.month == 12 else dt.date(cur.year, cur.month + 1, 1)
        key = (cur.year, cur.month)
        if key not in have_months:
            d1 = max(cur, start)
            d2 = min(nxt - dt.timedelta(days=1), end)
            yield d1, d2
        cur = nxt

# -------------------- upsert (FIXED: date & NaN) --------------------
def _to_record(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a pandas row dict to a JSON-serializable record for Supabase."""
    try:
        # dt (accept 'dt' or 'date')
        dt_raw = row.get("dt", row.get("date"))
        dti = pd.to_datetime(dt_raw, errors="coerce")
        if pd.isna(dti):
            return None
        dt_iso = dti.date().isoformat()

        def fnum(x):
            return None if pd.isna(x) else float(x)

        rec = {
            "exchange": "" if pd.isna(row.get("exchange")) else str(row.get("exchange")),
            "entity":   None if pd.isna(row.get("entity"))   else str(row.get("entity")),
            "k":        "" if pd.isna(row.get("k"))        else str(row.get("k")),
            "dt":       dt_iso,
            "close":    fnum(row.get("close")),
            "volume":   fnum(row.get("volume")),
        }
        return rec
    except Exception as e:
        _log(f"row serialization error: {e}")
        return None

def sb_upsert_trades(df: pd.DataFrame, batch_size: int = 800) -> int:
    """
    Upsert rows into public.trades in batches.
    Ensures JSON-serializable payload:
      - dt -> 'YYYY-MM-DD' (string)
      - close/volume -> float or None
      - exchange/entity/k -> strings
    Returns total rows successfully sent (best-effort).
    """
    if not SB_ENABLED:
        _log("POST upsert skipped: Supabase disabled")
        return 0
    if df is None or df.empty:
        _log("POST upsert skipped: empty dataframe")
        return 0

    # Minimal frame with expected columns
    cols = ["dt", "date", "close", "volume", "exchange", "entity", "k"]
    use = [c for c in cols if c in df.columns]
    payload = df[use].to_dict(orient="records")

    # Build serializable records
    records: list[Dict[str, Any]] = []
    for r in payload:
        rec = _to_record(r)
        if rec is not None:
            records.append(rec)

    if not records:
        _log("POST upsert skipped: nothing to send after serialization")
        return 0

    sent_total = 0
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        try:
            r = requests.post(
                f"{SUPABASE_URL}/rest/v1/trades?on_conflict=k,dt",
                headers={**HEADERS_JSON, "Prefer": "resolution=merge-duplicates,return=minimal"},
                data=json.dumps(chunk).encode("utf-8"),
                timeout=60,
            )
            if r.status_code not in (200, 201, 204):
                _log(f"POST upsert FAILED {r.status_code}: {r.text[:200]}")
                break
            sent_total += len(chunk)
            _log(f"POST upsert ✓ {len(chunk)} rows (total {sent_total})")
        except Exception as e:
            _log(f"POST upsert exception: {e}")
            break
    return sent_total

# -------------------- reads --------------------
def sb_load_range(k: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Load dt, close, volume for a key in [start, end] (inclusive)."""
    if not SB_ENABLED:
        return pd.DataFrame(columns=["dt", "close", "volume"])
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades"
            f"?select=dt,close,volume&k=eq.{k}&dt=gte.{start}&dt=lte.{end}&order=dt.asc",
            headers=HEADERS_JSON,
            timeout=60,
        )
        r.raise_for_status()
        j = r.json() or []
        _log(f"GET range {k} -> {len(j)} rows")
        if not j:
            return pd.DataFrame(columns=["dt", "close", "volume"])
        df = pd.DataFrame(j)
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.date
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        return df[["dt", "close", "volume"]].dropna(subset=["dt"])
    except Exception as e:
        _log(f"GET range error {k}: {e}")
        return pd.DataFrame(columns=["dt", "close", "volume"])

def sb_count_for_key(k: str) -> int:
    if not SB_ENABLED:
        return 0
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades?select=dt&k=eq.{k}&limit=1",
            headers={**HEADERS_JSON, "Prefer": "count=exact"},
            timeout=20,
        )
        total = (r.headers.get("content-range", "*/0").split("/")[-1]).strip()
        return int(total or 0)
    except Exception:
        return 0

def sb_max_date_for_key(k: str) -> Optional[dt.date]:
    """Latest dt stored for key k, or None if none."""
    if not SB_ENABLED:
        return None
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades?select=dt&k=eq.{k}&order=dt.desc&limit=1",
            headers=HEADERS_JSON,
            timeout=20,
        )
        r.raise_for_status()
        j = r.json() or []
        if not j:
            return None
        return pd.to_datetime(j[0].get("dt"), errors="coerce").date()
    except Exception as e:
        _log(f"max date error {k}: {e}")
        return None
