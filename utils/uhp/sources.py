# utils/uhp/sources.py
"""Unit Holding Pattern filings for the page.

Read-only view of what jobs/ingest_uhp.py published to the data branch. The page never
calls NSE or BSE itself (both refuse or throttle requests from cloud hosts).
"""
from __future__ import annotations

from utils.datastore import DataUnavailable, load_manifest, load_uhp_filings, load_uhp_xbrl, _load_uhp_filings  # noqa: F401


def fetch_master(index: str) -> tuple[list[dict], list[str]]:
    """(filing records for "reits" | "invits", problems to show the user)."""
    df = load_uhp_filings()
    problems: list[str] = []
    errors = load_manifest().get("uhp", {}).get("errors", [])
    if errors:
        problems.append(f"The last data update reported {len(errors)} error(s); some recent filings may be missing.")
    return df[df["index"] == index].to_dict("records"), problems


def fetch_xbrl(url_or_name: str) -> str:
    """XBRL text for a filing, given its original URL or stored file name."""
    return load_uhp_xbrl((url_or_name or "").rstrip("/").split("/")[-1])


def data_status() -> str:
    meta = load_manifest().get("uhp")
    if not meta:
        return ""
    return (
        f"Filings as of {meta['generated_at'][:10]} · latest as-on date {meta.get('latest_as_on') or '—'} · "
        f"{meta['filings']} filings for {meta['entities']} trusts"
    )


def clear_caches() -> None:
    load_manifest.clear()
    _load_uhp_filings.clear()
