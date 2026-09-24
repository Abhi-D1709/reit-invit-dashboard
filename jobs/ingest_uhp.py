# jobs/ingest_uhp.py
"""Collect SEBI Unit Holding Pattern (UHP) filings for REITs/InvITs.

    python -m jobs.ingest_uhp

Sources: NSE's corporate-unit-holdings master feed (all trusts listed on NSE) and, for
trusts listed only on BSE, the BSE unit-holding archive. Both list filings whose XBRL
documents are downloaded once and kept.

Output (under --data-dir, default ./data):
    uhp/filings.parquet   one row per (trust, as-on date): the latest filing
    uhp/xbrl/<file>.xml.gz  the XBRL document of every listed filing (immutable, gzipped)
    manifest.json         gets a "uhp" section (as-of time, counts, errors)

Rules (same spirit as the other jobs):
  * Failures are errors, never silently "nothing".
  * Records are merged into what is already stored, so a shorter or failed feed can never
    shrink the history. NSE's default feed shows only the latest 100 InvIT filings, so an
    explicit date range is always requested.
  * A filing with an XBRL document is listed only if that file is stored (the app parses it
    on demand). Older NSE filings often have no XBRL at all (the feed says ".../null"); they
    are kept with an empty xbrlFile, since their sponsor/public percentages still feed trends.
  * Only the latest submission per (trust, as-on date) is kept; BSE and NSE both re-file.
  * The entities sheet is the master list: only trusts in it are kept (a delisted trust, or one
    that surrendered its registration, is removed from the sheet and then from here).
  * BSE's archive API answers 403 to cloud hosts (GitHub Actions, Streamlit Cloud). That is
    reported as a warning, not an error, and the manifest records when BSE-only trusts were
    last refreshed. They file only ~4 times a year: run this job from a normal connection
    (`python -m jobs.ingest_uhp`, then commit/push data/) to refresh them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from utils.constants import DATA_DIR_NAME, ENTITIES_SHEET_CSV, HISTORY_START
from utils.uhp.xbrl_parser import parse_uhp_xbrl

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
NSE_HOME = "https://www.nseindia.com"
BSE_HOST = "https://www.bseindia.com"
BSE_ARCHIVE = "https://api.bseindia.com/BseIndiaAPI/api/unitholdingarchive/w?scripcode={scrip}"
BSE_HEADERS = {"User-Agent": UA, "Referer": BSE_HOST + "/", "Origin": BSE_HOST, "Accept": "application/json, text/plain, */*"}

# Used only if the entities sheet can't be read (scrip code, name, index).
FALLBACK_BSE_ONLY = [
    (542543, "Energy Infrastructure Trust", "invits"),
    (543225, "Altius Telecom Infrastructure Trust", "invits"),
    (543859, "Digital Fibre Infrastructure Trust", "invits"),
    (543925, "Maple Infrastructure Trust", "invits"),
    (544005, "Intelligent Supply Chain Infrastructure Trust", "invits"),
]

COLUMNS = [
    "index", "source", "ndsSymbol", "secLname", "secSname", "asOnDate", "submissionDate", "filedAt",
    "sponsorGroupPer", "publicHoldingPer", "xbrlFilePath", "xbrlFile", "bseScripCode", "ndsID",
]

_tls = threading.local()


def log(msg: str) -> None:
    print(msg, flush=True)


def _session(kind: str) -> requests.Session:
    """One session per thread and source (NSE needs its home-page cookie)."""
    attr = f"s_{kind}"
    if not hasattr(_tls, attr):
        s = requests.Session()
        s.headers.update(BSE_HEADERS if kind == "bse" else {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
        if kind == "nse":
            s.get(NSE_HOME + "/", timeout=20)
        setattr(_tls, attr, s)
    return getattr(_tls, attr)


def _get(kind: str, url: str, retries: int = 4) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = _session(kind).get(url, timeout=40)
            if r.status_code in (401, 403) and kind == "nse":
                delattr(_tls, "s_nse")  # rebuild the session (fresh cookie)
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            if not r.content:
                raise ValueError("empty response")
            return r
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{kind.upper()} {url}: {last}")


def _xbrl_path(xbrl_dir: Path, fname: str) -> Path:
    return xbrl_dir / (fname + ".gz")


def _read_xbrl(xbrl_dir: Path, fname: str) -> str:
    return gzip.decompress(_xbrl_path(xbrl_dir, fname).read_bytes()).decode("utf-8")


def _write_xbrl(xbrl_dir: Path, fname: str, text: str) -> None:
    # mtime=0 keeps the bytes deterministic, so re-runs never create spurious git changes
    _xbrl_path(xbrl_dir, fname).write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))


def _fmt(d: dt.datetime | dt.date) -> str:
    return d.strftime("%d-%b-%Y").upper()


# ------------------------------- NSE -----------------------------------------
def nse_records(index: str, start: dt.date, end: dt.date) -> list[dict]:
    """Filings from NSE's master feed. The feed's result varies with the query and even with
    the time of day (a plain request returns only the latest 100 InvIT filings, and the
    same dated request has returned 318 and 330 minutes apart), so several variants are
    requested and unioned."""
    base = NSE_HOME + "/api/corporate-unit-holdings-master?index=" + index
    d1 = start.strftime("%d-%m-%Y")
    queries = [
        f"{base}&from_date={d1}&to_date={end.strftime('%d-%m-%Y')}",
        f"{base}&from_date={d1}&to_date={(end + dt.timedelta(days=30)).strftime('%d-%m-%Y')}",
        base,
    ]
    data: list[dict] = []
    for i, url in enumerate(queries):
        try:
            part = _get("nse", url).json()
        except RuntimeError:
            if i == 0:
                raise  # the primary query must succeed
            continue
        if isinstance(part, list):
            data += part
    if not data:
        raise RuntimeError(f"NSE {index}: master feed returned no filings")
    out = []
    for r in data:
        try:
            filed = dt.datetime.strptime(r["broadCastDate"].title(), "%d-%b-%Y %H:%M:%S")
        except (KeyError, ValueError, AttributeError):
            filed = dt.datetime.strptime(r["submissionDate"].title(), "%d-%b-%Y")
        path = r.get("xbrlFilePath") or ""
        has_xbrl = bool(path) and not path.rstrip("/").lower().endswith("/null")
        out.append(
            {
                "index": index, "source": "NSE", "ndsSymbol": r["ndsSymbol"], "secLname": r["secLname"], "secSname": r.get("secSname") or r["secLname"],
                "asOnDate": r["asOnDate"].upper(), "submissionDate": r["submissionDate"].upper(), "filedAt": filed.isoformat(),
                "sponsorGroupPer": pd.to_numeric(r.get("sponsorGroupPer"), errors="coerce"),
                "publicHoldingPer": pd.to_numeric(r.get("publicHoldingPer"), errors="coerce"),
                "xbrlFilePath": path if has_xbrl else "", "xbrlFile": path.rstrip("/").split("/")[-1] if has_xbrl else "",
                "bseScripCode": "", "ndsID": str(r.get("ndsID", "")),
            }
        )
    return out


# ------------------------------- BSE -----------------------------------------
def load_tracked() -> tuple[set[str] | None, list[tuple[int, str, str]], str | None]:
    """(NSE symbols in the entities sheet, BSE-only trusts, warning). The NSE set is None when
    the sheet can't be read; nothing is then filtered out (better stale than wrongly deleted)."""
    try:
        last: Exception | None = None
        for attempt in range(4):  # Google's export endpoint times out now and then
            try:
                r = requests.get(ENTITIES_SHEET_CSV, headers={"User-Agent": UA}, timeout=45)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                last = e
                time.sleep(2 * (attempt + 1))
        else:
            raise last  # type: ignore[misc]
        df = pd.read_csv(io.StringIO(r.text), dtype=str).fillna("")
        need = {"Type of Entity", "Name of Entity", "NSE Symbol", "BSE Scrip Code"}
        if not need.issubset(df.columns):
            raise ValueError("entities sheet columns changed")
    except Exception as e:
        return None, FALLBACK_BSE_ONLY, f"Entities sheet unavailable ({e}); nothing was filtered and the built-in BSE-only list was used."
    nse = {x.strip().upper() for x in df["NSE Symbol"] if x.strip()}
    bse_only = []
    for _, r in df.iterrows():
        scrip = re.sub(r"[^0-9]", "", r["BSE Scrip Code"].split(".")[0])
        kind = r["Type of Entity"].strip().upper()
        if r["NSE Symbol"].strip() or not scrip or kind not in {"REIT", "INVIT"}:
            continue
        bse_only.append((int(scrip), r["Name of Entity"].strip(), "reits" if kind == "REIT" else "invits"))
    return nse, bse_only, None


def _bse_record(scrip: int, name: str, index: str, symbol: str, row: dict, xbrl_dir: Path) -> dict:
    url = BSE_HOST + row["XBRL_Link"]
    fname = url.rstrip("/").split("/")[-1]
    if _xbrl_path(xbrl_dir, fname).exists():
        text = _read_xbrl(xbrl_dir, fname)
    else:
        text = _get("bse", url).content.decode("utf-8-sig")
        _write_xbrl(xbrl_dir, fname, text)
    parsed = parse_uhp_xbrl(text)
    as_on = parsed.get("OneI", "DateOfReport") or parsed.contexts.get("OneI", {}).get("instant")
    if not as_on:
        raise RuntimeError(f"BSE {name}: no as-on date in {fname}")
    filed = row["Filing_Date_Time"]
    return {
        "index": index, "source": "BSE", "ndsSymbol": symbol, "secLname": name, "secSname": name,
        "asOnDate": _fmt(dt.datetime.fromisoformat(as_on[:10])), "submissionDate": _fmt(dt.datetime.fromisoformat(filed[:10])),
        "filedAt": dt.datetime.fromisoformat(filed).isoformat(),
        "sponsorGroupPer": pd.to_numeric(parsed.get("UnitHoldingOfSponsorAndSponsorGroupI", "AsAPercentageOfTotalOutStandingUnits"), errors="coerce"),
        "publicHoldingPer": pd.to_numeric(parsed.get("PublicHoldingI", "AsAPercentageOfTotalOutStandingUnits"), errors="coerce"),
        "xbrlFilePath": url, "xbrlFile": fname, "bseScripCode": str(scrip), "ndsID": f"BSE{scrip}-{row['Quarter']}",
    }


def bse_records(entities, xbrl_dir: Path, known_files: set[str], errors: list[str], blocked: list[str]) -> tuple[list[dict], int]:
    """(records, number of trusts whose archive was read successfully)."""
    jobs, reached = [], 0
    for scrip, name, index in entities:
        try:
            rows = _get("bse", BSE_ARCHIVE.format(scrip=scrip)).json().get("Table", [])
            reached += 1
        except (RuntimeError, ValueError) as e:
            (blocked if "403" in str(e) else errors).append(f"{name} ({scrip}): {e}")
            continue
        newest: dict[str, dict] = {}  # BSE lists every re-filing; only the latest per quarter is worth downloading
        for row in rows:
            if row.get("XBRL_Link"):
                q = str(row.get("qtr") or row.get("Quarter"))
                if q not in newest or row["Filing_Date_Time"] >= newest[q]["Filing_Date_Time"]:
                    newest[q] = row
        for row in newest.values():
            parts = (row.get("navigateurl") or "").split("/")
            symbol = parts[3].upper() if len(parts) > 3 and parts[3] else f"BSE{scrip}"
            jobs.append((scrip, name, index, symbol, row))

    def one(j):
        try:
            return _bse_record(*j, xbrl_dir), None
        except Exception as e:
            return None, f"BSE {j[1]} ({j[4].get('qtr', j[4].get('Quarter'))}): {e}"

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(one, jobs))
    out = []
    for rec, err in results:
        if err:
            errors.append(err)
        else:
            out.append(rec)
    return out, reached


# ------------------------------- shared --------------------------------------
def download_nse_xbrl(records: list[dict], xbrl_dir: Path, errors: list[str]) -> list[dict]:
    """Ensure every NSE filing's XBRL is stored; drop (and report) any that can't be fetched."""
    def one(rec):
        if not rec["xbrlFile"]:
            return rec, None  # no XBRL published for this filing
        if _xbrl_path(xbrl_dir, rec["xbrlFile"]).exists():
            return rec, None
        try:
            _write_xbrl(xbrl_dir, rec["xbrlFile"], _get("nse", rec["xbrlFilePath"]).text)
            return rec, None
        except Exception as e:
            return None, f"NSE {rec['ndsSymbol']} {rec['asOnDate']}: {e}"

    todo = [r for r in records if r["xbrlFile"] and not _xbrl_path(xbrl_dir, r["xbrlFile"]).exists()]
    if todo:
        log(f"  downloading {len(todo)} NSE XBRL file(s) ...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, records))
    ok = []
    for rec, err in results:
        if err:
            errors.append(err)
        else:
            ok.append(rec)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / DATA_DIR_NAME))
    ap.add_argument("--start", default=HISTORY_START)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    xbrl_dir = data_dir / "uhp" / "xbrl"
    xbrl_dir.mkdir(parents=True, exist_ok=True)
    fpath = data_dir / "uhp" / "filings.parquet"
    mpath = data_dir / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    existing = pd.read_parquet(fpath) if fpath.exists() else pd.DataFrame(columns=COLUMNS)
    start, end = dt.date.fromisoformat(args.start), dt.date.today()
    errors: list[str] = []
    warnings: list[str] = []

    tracked_nse, entities, warn = load_tracked()
    if warn:
        warnings.append(warn)

    fresh: list[dict] = []
    for index in ("reits", "invits"):
        log(f"NSE {index} ...")
        try:
            recs = nse_records(index, start, end)
            log(f"  {len(recs)} feed records (variants combined, before de-duplication)")
            if tracked_nse is not None:
                recs = [r for r in recs if r["ndsSymbol"] in tracked_nse]  # only trusts in the entities sheet
            latest: dict[tuple[str, str], dict] = {}  # only the newest submission per (trust, as-on date) is worth downloading
            for r in recs:
                k = (r["ndsSymbol"], r["asOnDate"])
                if k not in latest or r["filedAt"] >= latest[k]["filedAt"]:
                    latest[k] = r
            fresh += download_nse_xbrl(list(latest.values()), xbrl_dir, errors)
        except RuntimeError as e:
            errors.append(str(e))

    log("BSE-only trusts ...")
    blocked: list[str] = []
    bse_recs, bse_reached = bse_records(entities, xbrl_dir, set(existing["xbrlFile"]) if len(existing) else set(), errors, blocked)
    fresh += bse_recs
    log(f"  {len(entities)} trusts, {len(bse_recs)} BSE filings read, {bse_reached} archive(s) reachable")
    if blocked:
        warnings.append(
            f"BSE blocked the archive request for {len(blocked)} of {len(entities)} BSE-only trust(s) (HTTP 403, usual for cloud hosts); "
            "their filings were not refreshed. Run `python -m jobs.ingest_uhp` from a normal connection to refresh them."
        )

    # merge into what is stored; latest submission per (trust, as-on date) wins
    merged = pd.concat([f for f in (existing, pd.DataFrame(fresh, columns=COLUMNS)) if not f.empty], ignore_index=True)
    merged["_asof"] = pd.to_datetime(merged["asOnDate"].str.title(), format="%d-%b-%Y", errors="coerce")
    merged = merged.dropna(subset=["_asof"]).sort_values("filedAt").drop_duplicates(["ndsSymbol", "asOnDate"], keep="last")
    # drop trusts that are no longer in the entities sheet (also removes previously stored ones)
    excluded: list[str] = []
    excluded_rows = 0
    if tracked_nse is not None:
        bse_scrips = {str(e[0]) for e in entities}
        keep = ((merged["source"] == "NSE") & merged["ndsSymbol"].isin(tracked_nse)) | ((merged["source"] == "BSE") & merged["bseScripCode"].isin(bse_scrips))
        excluded = sorted(set(merged.loc[~keep, "ndsSymbol"]))
        excluded_rows = int((~keep).sum())
        merged = merged[keep]
        if excluded:
            log(f"  excluded (not in the entities sheet): {', '.join(excluded)}")
    # only list filings whose XBRL is on disk
    merged = merged[merged["xbrlFile"].map(lambda f: f == "" or _xbrl_path(xbrl_dir, f).exists())]
    merged = merged.sort_values(["index", "ndsSymbol", "_asof"], ascending=[True, True, False]).reset_index(drop=True)
    # trusts get renamed / re-cased over time; use the most recent name everywhere so a trust is one entity
    latest = merged.groupby("ndsSymbol").head(1).set_index("ndsSymbol")
    merged["secLname"] = merged["ndsSymbol"].map(latest["secLname"])
    merged["secSname"] = merged["ndsSymbol"].map(latest["secSname"])
    merged = merged.drop(columns="_asof")
    merged = merged[COLUMNS]

    # remove XBRL files no filing refers to any more (excluded trusts, superseded re-filings)
    referenced = {_xbrl_path(xbrl_dir, f).name for f in merged["xbrlFile"] if f}
    removed = 0
    for path in xbrl_dir.glob("*.gz"):
        if path.name not in referenced:
            path.unlink()
            removed += 1
    if removed:
        log(f"  removed {removed} unreferenced XBRL file(s)")

    if not merged.empty and not (fpath.exists() and pd.read_parquet(fpath).equals(merged)):
        merged.to_parquet(fpath, index=False, compression="zstd")
    if len(existing) and len(merged) < len(existing) - excluded_rows:
        warnings.append(f"Stored filings fell from {len(existing)} to {len(merged)} beyond the {excluded_rows} excluded (a re-filing was replaced, or files went missing).")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prev_bse = manifest.get("uhp", {}).get("bse_last_refreshed")
    manifest["uhp"] = {
        "bse_last_refreshed": now if (bse_reached == len(entities) and entities) else prev_bse,
        "generated_at": now,
        "filings": int(len(merged)),
        "entities": int(merged["ndsSymbol"].nunique()),
        "nse_filings": int((merged["source"] == "NSE").sum()),
        "bse_filings": int((merged["source"] == "BSE").sum()),
        "without_xbrl": int((merged["xbrlFile"] == "").sum()),
        "latest_as_on": merged["asOnDate"].map(lambda s: dt.datetime.strptime(s.title(), "%d-%b-%Y")).max().date().isoformat() if len(merged) else None,
        "excluded_symbols": excluded,
        "warnings": warnings,
        "errors": errors,
    }
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    log(f"\n{len(merged)} filings for {merged['ndsSymbol'].nunique()} trusts ({manifest['uhp']['nse_filings']} NSE, {manifest['uhp']['bse_filings']} BSE; {manifest['uhp']['without_xbrl']} without XBRL).")
    for w in warnings:
        log(f"  WARNING: {w}")
    for e in errors:
        log(f"  ERROR: {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
