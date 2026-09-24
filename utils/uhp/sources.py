# utils/uhp/sources.py
"""Unit Holding Pattern filings for REITs/InvITs: NSE's master feed plus the
BSE archive for trusts that are listed only on BSE (NSE's feed doesn't cover
them). Both are shaped like NSE master records so the UI treats them alike.

BSE-only trusts are derived from the entities sheet the Trading page already
uses (BSE Scrip Code present, NSE Symbol blank) rather than a hand-kept list.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from utils.common import ENTITIES_SHEET_CSV
from utils.uhp.xbrl_parser import parse_uhp_xbrl

# ------------------------------- config --------------------------------------
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

NSE_HOST = "https://www.nseindia.com"
NSE_MASTER_URL = NSE_HOST + "/api/corporate-unit-holdings-master?index={index}"
NSE_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

BSE_HOST = "https://www.bseindia.com"
BSE_ARCHIVE_URL = "https://api.bseindia.com/BseIndiaAPI/api/unitholdingarchive/w?scripcode={scrip}"
BSE_HEADERS = {
    "User-Agent": _UA,
    "Referer": BSE_HOST + "/",
    "Origin": BSE_HOST,
    "Accept": "application/json, text/plain, */*",
}

# Used only if the entities sheet can't be read (scrip code, name, "reits" | "invits").
_FALLBACK_BSE_ONLY = [
    (542543, "Energy Infrastructure Trust", "invits"),
    (543225, "Altius Telecom Infrastructure Trust", "invits"),
    (543859, "Digital Fibre Infrastructure Trust", "invits"),
    (543925, "Maple Infrastructure Trust", "invits"),
    (544005, "Intelligent Supply Chain Infrastructure Trust", "invits"),
]

# Filings are immutable, so parsed-source XML is kept on disk. The directory may
# be read-only or wiped on redeploy (e.g. Streamlit Cloud); the cache is optional.
CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


def _cache_read(name: str) -> str | None:
    try:
        p = CACHE_DIR / name
        return p.read_text(encoding="utf-8") if p.exists() else None
    except OSError:
        return None


def _cache_write(name: str, text: str) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / name).write_text(text, encoding="utf-8")
    except OSError:
        pass


# ------------------------------- NSE -----------------------------------------
@st.cache_resource(show_spinner=False)
def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get(NSE_HOST + "/", timeout=10)  # NSE requires a cookie from the home page
    except requests.RequestException:
        pass
    return s


def _nse_get(url: str) -> requests.Response:
    resp = _nse_session().get(url, timeout=20)
    if resp.status_code in (401, 403):  # cookie expired / blocked: rebuild session once
        _nse_session.clear()
        resp = _nse_session().get(url, timeout=20)
    resp.raise_for_status()
    return resp


@st.cache_data(ttl=3600, show_spinner="Fetching filing list from NSE...")
def _nse_master(index: str) -> list[dict]:
    return _nse_get(NSE_MASTER_URL.format(index=index)).json()


def _nse_xbrl(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    cached = _cache_read(name)
    if cached is not None:
        return cached
    text = _nse_get(url).text
    _cache_write(name, text)
    return text


# ------------------------------- BSE -----------------------------------------
_bse_session = requests.Session()
_bse_session.headers.update(BSE_HEADERS)


def _bse_get(url: str, retries: int = 3) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _bse_session.get(url, timeout=25)
            resp.raise_for_status()
            if resp.content:
                return resp
            last = RuntimeError("empty response")
        except requests.RequestException as e:
            last = e
        time.sleep(1 + attempt)
    raise RuntimeError(f"BSE request failed: {url}") from last


def _bse_xbrl(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    cached = _cache_read(name)
    if cached is not None:
        return cached
    text = _bse_get(url).content.decode("utf-8-sig")
    _cache_write(name, text)
    return text


def _fmt_date(iso: str) -> str:
    return datetime.fromisoformat(iso[:10]).strftime("%d-%b-%Y").upper()


@st.cache_data(ttl=1800, show_spinner=False)
def _bse_only_entities() -> tuple[list[tuple[int, str, str]], str | None]:
    """(entities, warning). Trusts with a BSE scrip code and no NSE symbol."""
    try:
        df = pd.read_csv(ENTITIES_SHEET_CSV, dtype=str, storage_options={"User-Agent": _UA})
    except Exception as e:  # network / sheet permissions / schema
        return _FALLBACK_BSE_ONLY, f"Entities sheet unavailable ({type(e).__name__}); using the built-in BSE-only list."

    need = {"Type of Entity", "Name of Entity", "NSE Symbol", "BSE Scrip Code"}
    if not need.issubset(df.columns):
        return _FALLBACK_BSE_ONLY, "Entities sheet columns changed; using the built-in BSE-only list."

    out: list[tuple[int, str, str]] = []
    for _, r in df.iterrows():
        nse = (r["NSE Symbol"] or "").strip() if isinstance(r["NSE Symbol"], str) else ""
        scrip = re.sub(r"\D", "", str(r["BSE Scrip Code"] or "").split(".")[0])
        kind = str(r["Type of Entity"] or "").strip().upper()
        if nse or not scrip or kind not in {"REIT", "INVIT"}:
            continue
        out.append((int(scrip), str(r["Name of Entity"]).strip(), "reits" if kind == "REIT" else "invits"))
    return out, None


def _bse_record(scrip: int, name: str, index: str, symbol: str, row: dict) -> dict | None:
    url = BSE_HOST + row["XBRL_Link"]
    try:
        parsed = parse_uhp_xbrl(_bse_xbrl(url))
    except Exception:
        return None
    as_on = parsed.get("OneI", "DateOfReport") or parsed.contexts.get("OneI", {}).get("instant")
    if not as_on:
        return None
    filed = row["Filing_Date_Time"]
    return {
        "asOnDate": _fmt_date(as_on),
        "broadCastDate": filed,
        "ndsID": f"BSE{scrip}-{row['Quarter']}",
        "ndsSymbol": symbol,
        "secLname": name,
        "secSname": name,
        "sponsorGroupPer": parsed.get("UnitHoldingOfSponsorAndSponsorGroupI", "AsAPercentageOfTotalOutStandingUnits"),
        "publicHoldingPer": parsed.get("PublicHoldingI", "AsAPercentageOfTotalOutStandingUnits"),
        "submissionDate": _fmt_date(filed),
        "xbrlFilePath": url,
        "source": "BSE",
        "index": index,
        "bseScripCode": str(scrip),
        "_filed": filed,
    }


@st.cache_data(ttl=3600, show_spinner="Fetching BSE-only filings...")
def _bse_records() -> tuple[list[dict], list[str], str | None]:
    """(records, failed filings, sheet warning). Each quarter can have several
    submissions on BSE; only the latest per (trust, as-on date) is kept."""
    entities, warn = _bse_only_entities()
    jobs = []
    for scrip, name, index in entities:
        rows = _bse_get(BSE_ARCHIVE_URL.format(scrip=scrip)).json().get("Table", [])
        for row in rows:
            if not row.get("XBRL_Link"):
                continue
            parts = (row.get("navigateurl") or "").split("/")
            symbol = parts[3].upper() if len(parts) > 3 and parts[3] else f"BSE{scrip}"
            jobs.append((scrip, name, index, symbol, row))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda j: _bse_record(*j), jobs))

    failed = [f"{j[1]} ({j[4].get('qtr', j[4].get('Quarter'))})" for j, r in zip(jobs, results) if r is None]

    latest: dict[tuple[str, str], dict] = {}
    for rec in filter(None, results):
        key = (rec["ndsSymbol"], rec["asOnDate"])
        if key not in latest or rec["_filed"] > latest[key]["_filed"]:
            latest[key] = rec
    return list(latest.values()), failed, warn


# ------------------------------- public API ----------------------------------
def clear_caches() -> None:
    _nse_master.clear()
    _bse_records.clear()
    _bse_only_entities.clear()


def fetch_master(index: str) -> tuple[list[dict], list[str]]:
    """Returns (records, problems). BSE trouble never blocks the NSE data; it is
    returned as messages for the UI to show."""
    records = [dict(r, source="NSE") for r in _nse_master(index)]
    problems: list[str] = []
    try:
        bse, failed, warn = _bse_records()
    except Exception as e:
        problems.append(f"BSE-only trusts could not be loaded ({type(e).__name__}); showing NSE-listed trusts only.")
    else:
        records += [r for r in bse if r["index"] == index]
        if warn:
            problems.append(warn)
        if failed:
            problems.append("Some BSE filings could not be loaded: " + ", ".join(failed))
    return records, problems


def fetch_xbrl(url: str) -> str:
    return _bse_xbrl(url) if url.startswith(BSE_HOST) else _nse_xbrl(url)
