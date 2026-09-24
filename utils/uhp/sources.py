# utils/uhp/sources.py
"""Unit Holding Pattern filings for the page.

Read-only view of what jobs/ingest_uhp.py published to the data branch. The page never
calls NSE or BSE itself (both refuse or throttle requests from cloud hosts).
"""
from __future__ import annotations

from datetime import datetime, timezone

from utils.datastore import DataUnavailable, load_manifest, load_uhp_filings, load_uhp_xbrl, _load_uhp_filings  # noqa: F401


BSE_STALE_DAYS = 45  # BSE-only trusts file quarterly; warn when the manual refresh is overdue


def fetch_master(index: str) -> tuple[list[dict], list[str]]:
    """(filing records for "reits" | "invits", problems to show the user)."""
    df = load_uhp_filings()
    problems: list[str] = []
    meta = load_manifest().get("uhp", {})
    if meta.get("errors"):
        problems.append(f"The last data update reported {len(meta['errors'])} error(s); some recent filings may be missing.")
    last_bse = meta.get("bse_last_refreshed")
    if index == "invits" and last_bse:
        age = (datetime.now(timezone.utc) - datetime.strptime(last_bse, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).days
        if age > BSE_STALE_DAYS:
            problems.append(
                f"Filings for trusts listed only on BSE were last refreshed {age} days ago "
                "(BSE blocks automated access from cloud hosts, so this is refreshed manually)."
            )
    return df[df["index"] == index].to_dict("records"), problems


def fetch_xbrl(url_or_name: str) -> str:
    """XBRL text for a filing, given its original URL or stored file name."""
    return load_uhp_xbrl((url_or_name or "").rstrip("/").split("/")[-1])


def data_status() -> str:
    meta = load_manifest().get("uhp")
    if not meta:
        return ""
    bse = f" · BSE-only trusts refreshed {meta['bse_last_refreshed'][:10]}" if meta.get("bse_last_refreshed") else ""
    return (
        f"Filings as of {meta['generated_at'][:10]} · latest as-on date {meta.get('latest_as_on') or '—'} · "
        f"{meta['filings']} filings for {meta['entities']} trusts{bse}"
    )


def clear_caches() -> None:
    load_manifest.clear()
    _load_uhp_filings.clear()
