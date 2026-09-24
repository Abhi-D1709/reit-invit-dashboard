# jobs/ingest_trades.py
"""Ingest NSE/BSE trading data for the tracked REITs/InvITs into monthly Parquet.

    python -m jobs.ingest_trades              # incremental (re-fetches a trailing window)
    python -m jobs.ingest_trades --full       # ignore checkpoints, rebuild from HISTORY_START

Output (under --data-dir, default ./data):
    trades/YYYY-MM.parquet   one file per month; only the current month changes daily
    manifest.json            as-of date, per-key coverage, checkpoints, warnings

Design rules (they exist because the old in-app fetcher broke each of them):
  * A failed request is an error, never "no data". Empty is only accepted when the
    source answered successfully with nothing (before listing, holiday, no trades).
  * A key's checkpoint advances only if every request for it succeeded.
  * Every run re-fetches a trailing window, so gaps heal on their own.
  * Rows are matched on BSE scrip code (ISIN is missing for some trusts).

Only needs requests, pandas and pyarrow. It does not import streamlit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from utils.constants import DATA_DIR_NAME, ENTITIES_SHEET_CSV, HISTORY_START

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
COLUMNS = ["key", "exchange", "symbol", "date", "close", "vwap", "volume", "turnover", "trades"]

NSE_HOME = "https://www.nseindia.com"
NSE_URL = (
    NSE_HOME + "/api/NextApi/apiClient/GetQuoteApi?functionName=getHistoricalTradeData"
    "&symbol={symbol}&series={series}&fromDate={d1}&toDate={d2}"
)
NSE_CHUNK_DAYS = 90  # NSE truncates long ranges; 90 days is known to return complete data

BSE_NEW_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{ymd}_F_0000.CSV"
BSE_OLD_URL = "https://www.bseindia.com/download/BhavCopy/Equity/EQ_ISINCODE_{dmy}.zip"
BSE_NEW_FROM = dt.date(2024, 4, 1)  # both formats exist around here; prefer the format native to the date


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------- entities ------------------------------------
def load_targets() -> tuple[list[tuple[str, str]], list[int]]:
    """(NSE (symbol, series) pairs, BSE scrip codes) from the entities sheet."""
    r = requests.get(ENTITIES_SHEET_CSV, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), dtype=str).fillna("")
    need = {"NSE Symbol", "NSE Series", "BSE Scrip Code"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"Entities sheet is missing columns: {sorted(need - set(df.columns))}")
    nse = sorted({(a.strip().upper(), b.strip().upper()) for a, b in zip(df["NSE Symbol"], df["NSE Series"]) if a.strip() and b.strip()})
    bse = sorted({int("".join(ch for ch in c.split(".")[0] if ch.isdigit())) for c in df["BSE Scrip Code"] if any(ch.isdigit() for ch in c)})
    return nse, bse


# ------------------------------- storage -------------------------------------
def load_existing(data_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted((data_dir / "trades").glob("*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(parts, ignore_index=True)


def save_months(df: pd.DataFrame, data_dir: Path) -> list[str]:
    """Write one Parquet per month; skip months whose content is unchanged (keeps git quiet)."""
    out_dir = data_dir / "trades"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["key", "date"]).reset_index(drop=True)
    ym = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    months = []
    for m, part in df.groupby(ym):
        part = part.reset_index(drop=True)
        path = out_dir / f"{m}.parquet"
        if path.exists() and pd.read_parquet(path).equals(part):
            months.append(m)
            continue
        part.to_parquet(path, index=False, compression="zstd")
        months.append(m)
    return months


def read_manifest(data_dir: Path) -> dict:
    p = data_dir / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------- NSE -----------------------------------------
def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Referer": NSE_HOME + "/", "Accept": "application/json, text/plain, */*"})
    s.get(NSE_HOME, timeout=20)  # cookie
    return s


def _nse_window(sess: requests.Session, symbol: str, series: str, d1: dt.date, d2: dt.date, retries: int = 4) -> list[dict]:
    url = NSE_URL.format(symbol=symbol, series=series, d1=d1.strftime("%d-%m-%Y"), d2=d2.strftime("%d-%m-%Y"))
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code in (401, 403):
                sess.get(NSE_HOME, timeout=20)  # refresh cookie
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            raise ValueError(f"unexpected payload: {str(data)[:80]}")
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"NSE {symbol}/{series} {d1}..{d2}: {last}")


def _nse_frame(rows: list[dict], symbol: str, series: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    r = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "key": f"NSE:{symbol}:{series}",
            "exchange": "NSE",
            "symbol": symbol,
            "date": pd.to_datetime(r["mtimestamp"], format="%d-%b-%Y", errors="coerce").dt.date,
            "close": pd.to_numeric(r["chClosingPrice"], errors="coerce"),
            "vwap": pd.to_numeric(r["vwap"], errors="coerce"),
            "volume": pd.to_numeric(r["chTotTradedQty"], errors="coerce"),
            "turnover": pd.to_numeric(r["chTotTradedVal"], errors="coerce"),
            "trades": pd.to_numeric(r["chTotalTrades"], errors="coerce"),
        }
    )
    return out.dropna(subset=["date"])


def _nse_symbol(symbol: str, series: str, start: dt.date, end: dt.date) -> tuple[pd.DataFrame, str | None]:
    """All rows for one symbol in [start, end]. Returns (rows, error). Rows from
    successful windows are kept even if a later window fails."""
    sess = _nse_session()
    frames, error = [], None
    d = start
    while d <= end:
        d2 = min(d + dt.timedelta(days=NSE_CHUNK_DAYS - 1), end)
        try:
            frames.append(_nse_frame(_nse_window(sess, symbol, series, d, d2), symbol, series))
        except RuntimeError as e:
            error = str(e)
            break
        time.sleep(0.25)
        d = d2 + dt.timedelta(days=1)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return df, error


def run_nse(targets, checkpoints: dict, start_default: dt.date, end: dt.date, refresh_days: int, full: bool):
    jobs = []
    for symbol, series in targets:
        key = f"NSE:{symbol}:{series}"
        done = checkpoints.get(key)
        start = start_default if (full or not done) else max(start_default, dt.date.fromisoformat(done) - dt.timedelta(days=refresh_days))
        jobs.append((symbol, series, key, start))

    frames, errors, advanced = [], [], {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_nse_symbol, s, se, st, end): (s, se, k, st) for s, se, k, st in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            s, se, k, st = futs[fut]
            try:
                df, err = fut.result()
            except Exception as e:  # never let one symbol kill the run
                df, err = pd.DataFrame(columns=COLUMNS), f"NSE {s}/{se}: {e}"
            frames.append(df)
            if err:
                errors.append(err)
            else:
                advanced[k] = end.isoformat()
            log(f"  NSE [{i}/{len(jobs)}] {s}/{se}: {len(df)} rows from {st}" + (f"  ERROR {err}" if err else ""))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)), errors, advanced


# ------------------------------- BSE -----------------------------------------
_tls = threading.local()


def _bse_session() -> requests.Session:
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
        _tls.s.headers.update({"User-Agent": UA})
    return _tls.s


def _bse_get(url: str, retries: int = 4) -> bytes | None:
    """Bytes of the file, or None if BSE has no file for that date.
    BSE answers a missing file with HTTP 200 + an HTML page, so content is checked."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = _bse_session().get(url, timeout=40)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            head = r.content[:64].lstrip().lower()
            if head.startswith((b"<!doctype", b"<html", b"<")) or len(r.content) < 500:
                return None
            return r.content
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"BSE {url}: {last}")


def _bse_parse_new(content: bytes, day: dt.date, scrips: set[int]) -> pd.DataFrame | None:
    df = pd.read_csv(io.BytesIO(content), usecols=["TradDt", "FinInstrmId", "ClsPric", "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd"])
    if (pd.to_datetime(df["TradDt"]).dt.date != day).all():
        return None  # file for a different day: treat as no session
    df = df[df["FinInstrmId"].isin(scrips)]
    return pd.DataFrame({"code": df["FinInstrmId"], "close": df["ClsPric"], "volume": df["TtlTradgVol"], "turnover": df["TtlTrfVal"], "trades": df["TtlNbOfTxsExctd"]})


def _bse_parse_old(content: bytes, day: dt.date, scrips: set[int]) -> pd.DataFrame | None:
    """Old EQ_ISINCODE zip. Layout drifted over the years: files before ~2018 have no
    TRADING_DATE column, and a few rows elsewhere are malformed (commas inside company
    names). So: read everything as text, coerce numerics, and only trust rows whose
    SC_CODE parses; verify the file's date only when it can be verified."""
    z = zipfile.ZipFile(io.BytesIO(content))
    wanted = {"SC_CODE", "CLOSE", "NO_OF_SHRS", "NET_TURNOV", "NO_TRADES", "TRADING_DATE"}
    df = pd.read_csv(z.open(z.namelist()[0]), dtype=str, usecols=lambda c: c in wanted, on_bad_lines="skip")
    if "TRADING_DATE" in df.columns:
        seen = pd.to_datetime(df["TRADING_DATE"], format="%d-%b-%y", errors="coerce").dt.date.dropna()
        if len(seen) and (seen != day).mean() > 0.5:
            return None  # file is for a different day: treat as no session
    df = df.assign(SC_CODE=pd.to_numeric(df["SC_CODE"], errors="coerce"))
    df = df[df["SC_CODE"].isin(scrips)]
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    return pd.DataFrame({"code": df["SC_CODE"], "close": num("CLOSE"), "volume": num("NO_OF_SHRS"), "turnover": num("NET_TURNOV"), "trades": num("NO_TRADES")})


def _bse_day(day: dt.date, scrips: set[int]) -> tuple[dt.date, pd.DataFrame | None]:
    """(day, rows) or (day, None) when BSE had no session/file that day."""
    new = (BSE_NEW_URL.format(ymd=day.strftime("%Y%m%d")), _bse_parse_new)
    old = (BSE_OLD_URL.format(dmy=day.strftime("%d%m%y")), _bse_parse_old)
    for url, parse in ([new, old] if day >= BSE_NEW_FROM else [old, new]):
        content = _bse_get(url)
        if content is None:
            continue
        rows = parse(content, day, scrips)
        if rows is not None:
            return day, rows
    return day, None


def run_bse(scrips: list[int], start: dt.date, end: dt.date, extra_dates: set[dt.date]):
    scrip_set = set(scrips)
    days, d = [], start
    while d <= end:
        if d.weekday() < 5 or d in extra_dates:  # weekends only if NSE traded (special sessions)
            days.append(d)
        d += dt.timedelta(days=1)

    frames, no_file, errors = [], [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_bse_day, day, scrip_set): day for day in days}
        for i, fut in enumerate(as_completed(futs), 1):
            day = futs[fut]
            try:
                _, rows = fut.result()
            except Exception as e:
                errors.append(f"BSE {day}: {e}")
                continue
            if rows is None:
                no_file.append(day)
            else:
                rows = rows.copy()
                rows["date"] = day
                frames.append(rows)
            if i % 200 == 0 or i == len(days):
                log(f"  BSE [{i}/{len(days)}] days processed")
    if not frames:
        return pd.DataFrame(columns=COLUMNS), sorted(no_file), errors
    r = pd.concat(frames, ignore_index=True)
    r["code"] = r["code"].astype(int)
    out = pd.DataFrame(
        {
            "key": "BSE:" + r["code"].astype(str),
            "exchange": "BSE",
            "symbol": r["code"].astype(str),
            "date": r["date"],
            "close": pd.to_numeric(r["close"], errors="coerce"),
            "vwap": (pd.to_numeric(r["turnover"], errors="coerce") / pd.to_numeric(r["volume"], errors="coerce")).round(4),
            "volume": pd.to_numeric(r["volume"], errors="coerce"),
            "turnover": pd.to_numeric(r["turnover"], errors="coerce"),
            "trades": pd.to_numeric(r["trades"], errors="coerce"),
        }
    )
    return out, sorted(no_file), errors


# ------------------------------- main ----------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / DATA_DIR_NAME))
    ap.add_argument("--start", default=HISTORY_START, help="earliest date to backfill from")
    ap.add_argument("--end", default=None, help="last date (default: today)")
    ap.add_argument("--refresh-days", type=int, default=15, help="re-fetch this many days before each checkpoint")
    ap.add_argument("--full", action="store_true", help="ignore checkpoints and refetch everything from --start")
    ap.add_argument("--only", choices=["nse", "bse"], default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    manifest = read_manifest(data_dir)
    keys_meta: dict = manifest.get("trades", {}).get("keys", {})
    checkpoints = {k: v["checked_through"] for k, v in keys_meta.items() if "checked_through" in v}

    log(f"Loading tracked entities from the sheet ...")
    nse_targets, bse_scrips = load_targets()
    log(f"  {len(nse_targets)} NSE symbols, {len(bse_scrips)} BSE scrips; window {start} .. {end}")

    existing = load_existing(data_dir)
    frames, errors, warnings = [existing], [], []
    advanced: dict[str, str] = {}

    if args.only != "bse":
        log("NSE ...")
        nse_df, nse_err, advanced = run_nse(nse_targets, checkpoints, start, end, args.refresh_days, args.full)
        frames.append(nse_df)
        errors += nse_err

    bse_no_file: list[dt.date] = []
    if args.only != "nse":
        log("BSE ...")
        bse_ck = manifest.get("bse", {}).get("checked_through")
        bse_start = start if (args.full or not bse_ck) else max(start, dt.date.fromisoformat(bse_ck) - dt.timedelta(days=args.refresh_days))
        nse_all = pd.concat([existing, frames[-1]] if args.only != "bse" else [existing], ignore_index=True)
        extra = {d for d in pd.to_datetime(nse_all[nse_all["exchange"] == "NSE"]["date"]).dt.date.unique() if d.weekday() >= 5}
        bse_df, bse_no_file, bse_err = run_bse(bse_scrips, bse_start, end, extra)
        frames.append(bse_df)
        errors += bse_err
        if not bse_err:
            manifest.setdefault("bse", {})["checked_through"] = end.isoformat()

    frames = [f for f in frames if not f.empty]
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged = merged.drop_duplicates(subset=["key", "date"], keep="last")[COLUMNS]

    # --- consistency warnings
    if args.only != "nse":
        nse_days = set(merged[merged["exchange"] == "NSE"]["date"])
        missing_bse = sorted(d for d in bse_no_file if d in nse_days and bse_start <= d)
        if missing_bse:
            warnings.append(f"BSE had no file on {len(missing_bse)} day(s) when NSE traded: {[d.isoformat() for d in missing_bse[:10]]}")
        known = set(manifest.get("bse", {}).get("no_file_dates", []))
        known = {d for d in known if not (bse_start.isoformat() <= d <= end.isoformat())}
        manifest.setdefault("bse", {})["no_file_dates"] = sorted(known | {d.isoformat() for d in bse_no_file})
    for symbol, series in nse_targets:
        if f"NSE:{symbol}:{series}" not in set(merged["key"]):
            warnings.append(f"NSE {symbol}/{series}: no rows at all (unlisted, no trades, or series changed)")
    for code in bse_scrips:
        if f"BSE:{code}" not in set(merged["key"]):
            warnings.append(f"BSE {code}: no rows at all")

    months = save_months(merged, data_dir)
    per_key = {}
    for k, g in merged.groupby("key"):
        meta = {"first": min(g["date"]).isoformat(), "last": max(g["date"]).isoformat(), "rows": int(len(g))}
        prev = keys_meta.get(k, {})
        meta["checked_through"] = advanced.get(k, prev.get("checked_through", meta["last"] if k.startswith("BSE:") else None))
        if meta["checked_through"] is None:
            del meta["checked_through"]
        per_key[k] = meta
    for k, v in advanced.items():  # keys with no rows yet still record that they were checked
        per_key.setdefault(k, {"first": None, "last": None, "rows": 0})["checked_through"] = v

    manifest.update(
        {
            "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "history_start": start.isoformat() if args.full else manifest.get("history_start", start.isoformat()),
            "trades": {
                "first_date": min(merged["date"]).isoformat(),
                "last_date": max(merged["date"]).isoformat(),
                "rows": int(len(merged)),
                "months": months,
                "keys": per_key,
            },
            "warnings": warnings,
            "errors": errors,
        }
    )
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    log(f"\nWrote {len(merged):,} rows across {len(months)} monthly files; last date {max(merged['date'])}.")
    for w in warnings:
        log(f"  WARNING: {w}")
    for e in errors:
        log(f"  ERROR: {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
