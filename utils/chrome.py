# utils/chrome.py
"""Page furniture shared by every page: the "data as of" line."""
from __future__ import annotations

import datetime as dt
from typing import Optional

import streamlit as st


def _day(iso: Optional[str]) -> Optional[str]:
    """'2026-09-24T09:40:14Z' or '2026-09-24' -> '24 Sep 2026'."""
    if not iso:
        return None
    try:
        return dt.date.fromisoformat(str(iso)[:10]).strftime("%d %b %Y").lstrip("0")
    except ValueError:
        return None


def data_as_of_text(manifest: dict) -> str:
    """One line saying how fresh each data source is, from the data branch's manifest."""
    parts = []
    trades = _day((manifest.get("trades") or {}).get("last_date"))
    if trades:
        parts.append(f"Trading data through {trades}")
    uhp = _day((manifest.get("uhp") or {}).get("generated_at"))
    if uhp:
        parts.append(f"Unit holding filings as of {uhp}")
    ibbi = _day((manifest.get("ibbi") or {}).get("generated_at"))
    if ibbi:
        parts.append(f"IBBI valuer registry as of {ibbi}")
    parts.append("Google Sheets are read live (cached for up to 10 minutes)")
    errors = len(manifest.get("errors") or []) + len((manifest.get("uhp") or {}).get("errors") or []) + len((manifest.get("ibbi") or {}).get("errors") or [])
    if errors:
        parts.append(f"{errors} data-job error(s) reported: some data may be out of date")
    return " · ".join(parts)


def render_data_banner() -> None:
    """Small, muted line at the top of every page. Never blocks the page if the manifest can't be read."""
    try:
        from utils.datastore import load_manifest

        st.caption(data_as_of_text(load_manifest()), help="Machine-collected data (trading, filings, registry) is refreshed by scheduled jobs.")
    except Exception:
        st.caption("Data freshness unavailable right now. Google Sheets are read live (cached for up to 10 minutes).")
