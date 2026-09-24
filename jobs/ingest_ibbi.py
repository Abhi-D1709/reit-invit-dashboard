# jobs/ingest_ibbi.py
"""Scrape the IBBI register of registered valuers into Parquet.

    python -m jobs.ingest_ibbi

Output (under --data-dir, default ./data):
    ibbi/individuals.parquet   reg_no, name, rvo, registration_date, asset_class, status, cancelled_on
    ibbi/entities.parquet      reg_no, name, constitution, rvo, asset_class, status, cancelled_on
    manifest.json              gets an "ibbi" section (as-of time, row counts)

Rules (same spirit as ingest_trades):
  * A failed request is an error, never an empty page.
  * Each register numbers its rows (S.No. 1..N). A scrape is accepted only if those
    numbers are contiguous, so a dropped page cannot silently shorten the registry.
  * Cancelled registrations appear on the site as short rows ("Registration Cancelled
    w.e.f. ..."). They are kept, with status/cancelled_on, not dropped.
  * A scrape that comes back much smaller than the stored one is rejected.
  * Personal contact details (addresses, emails, directors) are deliberately NOT
    stored: the app matches on registration number and name only, and this repo is public.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.constants import DATA_DIR_NAME

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/xhtml+xml"}
MIN_KEEP_RATIO = 0.95  # refuse to overwrite with a scrape smaller than this share of the stored one

# kind -> (url template, minimum cells per row, {output column: cell index})
REGISTERS = {
    "individuals": (
        "https://ibbi.gov.in/service-provider/rvs?page={page}",
        8,
        {"reg_no": 1, "name": 2, "rvo": 5, "registration_date": 6, "asset_class": 7},
    ),
    "entities": (
        "https://ibbi.gov.in/service-provider/rvo-entities?page={page}",
        9,
        {"reg_no": 1, "constitution": 2, "name": 3, "rvo": 6, "asset_class": 8},
    ),
}


CANCEL_RE = re.compile(r"cancel|suspend|withdraw|surrender", re.I)
DATE_RE = re.compile(r"(\d{1,2}\s+[A-Za-z]{3},?\s+\d{4})")


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_page(kind: str, page: int, retries: int = 4) -> list[list[str]] | None:
    """Rows (list of cell texts) for a page; None if the page is past the end.
    Network/HTTP failures raise, they never look like an empty page."""
    url, _, _ = REGISTERS[kind]
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url.format(page=page), headers=HEADERS, timeout=40)
            if r.status_code == 404:
                return None  # IBBI answers 404 beyond the last page of a register
            r.raise_for_status()
            table = BeautifulSoup(r.text, "html.parser").find("table", class_="reporttable")
            body = table.find("tbody") if table else None
            if body is None:
                if table is None and "reporttable" not in r.text:
                    raise ValueError("register table not found in page (layout change or block page)")
                return None
            rows = [[td.get_text(strip=True) for td in tr.find_all("td")] for tr in body.find_all("tr")]
            return rows or None
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"IBBI {kind} page {page}: {last}")


def find_last_page(kind: str) -> int:
    """Highest page that has rows (exponential probe, then binary search)."""
    if fetch_page(kind, 1) is None:
        raise RuntimeError(f"IBBI {kind}: page 1 is empty; refusing to continue")
    lo, hi = 1, 2
    while fetch_page(kind, hi) is not None:
        lo, hi = hi, hi * 2
        if hi > 20000:
            raise RuntimeError(f"IBBI {kind}: no end found up to page {hi}")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fetch_page(kind, mid) is not None:
            lo = mid
        else:
            hi = mid
    return lo


def scrape(kind: str, workers: int) -> pd.DataFrame:
    _, min_cells, cols = REGISTERS[kind]
    last = find_last_page(kind)
    log(f"  {kind}: {last} pages")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pages = list(pool.map(lambda p: (p, fetch_page(kind, p)), range(1, last + 1)))

    records, problems = [], []
    for p, rows in pages:
        if rows is None:
            problems.append(f"page {p} returned no rows")
            continue
        for cells in rows:
            if len(cells) >= min_cells:
                data, status, cancelled_on = cells, "Registered", ""
            elif len(cells) >= 3 and CANCEL_RE.search(cells[-1]):
                # e.g. [S.No., reg no, name, "Registration Cancelled w.e.f. 09 Feb, 2021"]
                data, status = cells[:-1], cells[-1].strip()
                m = DATE_RE.search(status)
                cancelled_on = pd.to_datetime(m.group(1).replace(",", ""), format="%d %b %Y", errors="coerce") if m else pd.NaT
                cancelled_on = "" if pd.isna(cancelled_on) else cancelled_on.date().isoformat()
            else:
                problems.append(f"page {p}: unexpected row layout {cells[:4]}")
                continue
            rec = {out: (data[i].strip() if i < len(data) else "") for out, i in cols.items()}
            rec.update(status=status, cancelled_on=cancelled_on, _sno=data[0].strip())
            records.append(rec)
    if problems:
        raise RuntimeError(f"IBBI {kind}: incomplete scrape, e.g. {problems[:3]}")

    df = pd.DataFrame(records)
    df["reg_no"] = df["reg_no"].str.upper()
    serials = pd.to_numeric(df["_sno"].str.replace(",", ""), errors="coerce")  # site prints 2,707
    if serials.isna().any() or sorted(serials.astype(int)) != list(range(1, len(df) + 1)):
        raise RuntimeError(f"IBBI {kind}: row numbers are not contiguous 1..{len(df)}; a page was likely missed")
    if (df["reg_no"] == "").any():
        raise RuntimeError(f"IBBI {kind}: rows with an empty registration number")
    dupes = int(df["reg_no"].duplicated().sum())
    if dupes:
        log(f"  {kind}: note: {dupes} duplicate registration numbers in the source; keeping the first of each")
        df = df.drop_duplicates("reg_no", keep="first")
    return df.drop(columns="_sno").sort_values("reg_no").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / DATA_DIR_NAME))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "ibbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = data_dir / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}

    errors, counts = [], {}
    for kind in REGISTERS:
        log(f"Scraping IBBI {kind} ...")
        path = out_dir / f"{kind}.parquet"
        try:
            df = scrape(kind, args.workers)
            if path.exists():
                prev = pd.read_parquet(path)
                if len(df) < MIN_KEEP_RATIO * len(prev):
                    raise RuntimeError(f"IBBI {kind}: scraped {len(df)} rows vs {len(prev)} stored; refusing to overwrite")
                if prev.equals(df):
                    counts[kind] = len(df)
                    log(f"  {kind}: {len(df)} rows, unchanged")
                    continue
            df.to_parquet(path, index=False, compression="zstd")
            counts[kind] = len(df)
            log(f"  {kind}: {len(df)} rows written")
        except Exception as e:
            errors.append(str(e))
            log(f"  ERROR {e}")
            if path.exists():
                counts[kind] = len(pd.read_parquet(path))  # keep serving the previous copy

    manifest["ibbi"] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "individuals": counts.get("individuals", 0),
        "entities": counts.get("entities", 0),
        "errors": errors,
    }
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
