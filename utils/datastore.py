# utils/datastore.py
"""Read-only access to the machine-generated data written by jobs/.

The data lives on the `data` git branch (public repo), so the deployed app
needs no credentials: it reads Parquet over HTTPS from raw.githubusercontent.com.
When a local ./data checkout exists (developer machine) that is used instead;
set DATA_SOURCE=remote to force the remote copy.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import streamlit as st

from utils.constants import DATA_BRANCH, DATA_DIR_NAME, DATA_REPO

LOCAL_DIR = Path(__file__).resolve().parents[1] / DATA_DIR_NAME
RAW_BASE = f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_BRANCH}"


class DataUnavailable(RuntimeError):
    """The data branch could not be read."""


def _use_local() -> bool:
    return os.getenv("DATA_SOURCE", "").lower() != "remote" and (LOCAL_DIR / "manifest.json").exists()


def source_label() -> str:
    return "local data/ checkout" if _use_local() else f"github.com/{DATA_REPO} ({DATA_BRANCH} branch)"


def _read_bytes(rel: str) -> bytes:
    if _use_local():
        return (LOCAL_DIR / rel).read_bytes()
    try:
        r = requests.get(f"{RAW_BASE}/{rel}", timeout=30)
        r.raise_for_status()
        return r.content
    except requests.RequestException as e:
        raise DataUnavailable(f"Could not read {rel} from the data branch: {e}") from e


@st.cache_data(ttl=900, show_spinner=False)
def load_manifest() -> dict:
    """Coverage, as-of date and warnings for the data branch (short cache so new data shows up quickly)."""
    return json.loads(_read_bytes("manifest.json"))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_month(ym: str, generated_at: str) -> pd.DataFrame:
    # generated_at is part of the cache key: a new data run invalidates old months
    return pd.read_parquet(io.BytesIO(_read_bytes(f"trades/{ym}.parquet")))


def load_trades(keys: Iterable[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Daily rows (key, exchange, symbol, date, close, vwap, volume, turnover, trades)
    for the given keys in [start, end]. Keys look like 'NSE:EMBASSY:RR' or 'BSE:542602'."""
    m = load_manifest()
    first_ym, last_ym = start.strftime("%Y-%m"), end.strftime("%Y-%m")
    months = [ym for ym in m["trades"]["months"] if first_ym <= ym <= last_ym]
    keys = set(keys)
    parts = [_load_month(ym, m["generated_at"]) for ym in months]
    parts = [p[p["key"].isin(keys)] for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["key", "exchange", "symbol", "date", "close", "vwap", "volume", "turnover", "trades"])
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df.sort_values(["key", "date"]).reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_ibbi_table(name: str, generated_at: str) -> pd.DataFrame:
    # generated_at is part of the cache key: a new scrape invalidates the cached copy
    return pd.read_parquet(io.BytesIO(_read_bytes(f"ibbi/{name}.parquet")))


def load_ibbi() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """IBBI register of registered valuers: (individuals, entities, meta).
    Columns: reg_no, name, rvo, asset_class, status ("Registered" or the cancellation note),
    cancelled_on; individuals also registration_date, entities also constitution."""
    meta = load_manifest().get("ibbi")
    if not meta:
        raise DataUnavailable("The IBBI registry has not been published to the data branch yet.")
    key = meta["generated_at"]
    return _load_ibbi_table("individuals", key), _load_ibbi_table("entities", key), meta
