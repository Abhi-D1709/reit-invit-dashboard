# utils/common.py
import io
import re
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

# ---------- Defaults the app uses ----------
DEFAULT_REIT_BORR_URL  = "https://docs.google.com/spreadsheets/d/1OugwmVbR2BXjWcRGOlLhqrg3APVv9R17LYpZPDeFDkw/edit?usp=sharing"
DEFAULT_INVIT_BORR_URL = "https://docs.google.com/spreadsheets/d/1Zqi5VWeS2GSfhWa0gVPruIHdYzYM6luWalOv_8mhHsc/edit?usp=sharing"

DEFAULT_REIT_FUND_URL  = "https://docs.google.com/spreadsheets/d/1cuH2odCdJpnP5E0trvroQWzB4rXFgWGpjhHcAuX81Hs/edit?usp=sharing"
DEFAULT_INVIT_FUND_URL = "https://docs.google.com/spreadsheets/d/1eepPHnjo31G3ueeQTGxmVT7iY9cXh3NRgpDJuq6ygS8/edit?usp=sharing"

# Basic_Details (REIT / InvIT) – used only by tabs/basic_details.py
DEFAULT_REIT_DIR_URL   = "https://docs.google.com/spreadsheets/d/1PnuNGHDskqBZt4WUO8JpmssaQ3nOUvLJolfwB7T5zDE/edit?usp=sharing"
DEFAULT_INVIT_DIR_URL  = "https://docs.google.com/spreadsheets/d/1twj3iCRDOk46Hb8xGHgkDCW5hfUp170EltXKx0aepm8/edit?usp=sharing"  # fill when your InvIT directory sheet is ready

# Trading entities (REIT/InvIT mapping) — you can change this sheet anytime
ENTITIES_SHEET_ID = "1g44Lkv3VZU4FDTzrWXKhdxGwrWecZHHrZLmuRMyFHDI"
ENTITIES_SHEET_CSV = f"https://docs.google.com/spreadsheets/d/{ENTITIES_SHEET_ID}/export?format=csv"

# Sponsor Holding (REIT) – Sheet3 (gid=1466135872) of your master sheet
DEFAULT_REIT_SPON_URL  = "https://docs.google.com/spreadsheets/d/1PnuNGHDskqBZt4WUO8JpmssaQ3nOUvLJolfwB7T5zDE/export?format=csv&gid=1466135872"

# NDCF (REITs) – Public view link
NDCF_REITS_SHEET_URL = "https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing"
NDCF_INVITS_SHEET_URL = "" #to be added when available

# Sponsor Holding (InvIT) – set this when your InvIT sheet is ready
DEFAULT_INVIT_SPON_URL = ""

#Governance Data (REITs) – Public view link
GOVERNANCE_REIT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ETx5UZKQQyZKxkF4fFJ4R9wa7i7TNp7EXIhHWiVYG7s/edit?usp=sharing"

# Valuation Data (REITs) – Public view link
VALUATION_REIT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1dujQ1XpnorGgXvmrlSjWuBkQFGifnC39sqbG98W9TrQ/edit?usp=sharing"

# Canonical internal names
ENT_COL = "__Entity__"
FY_COL  = "__FinancialYear__"
QTR_COL = "__QuarterEnded__"

AAA_PAT = re.compile(r'(^|\W)(AAA|Aaa)($|\W)', re.I)
EPS = 1e-9

# ---------- Styling ----------
def inject_global_css():
    st.markdown(
        """
        <style>
          .app-hero {
            padding: 14px 18px;
            border-radius: 14px;
            border: 1px solid rgba(0,0,0,0.06);
            background: linear-gradient(180deg, rgba(25,118,210,0.08) 0%, rgba(25,118,210,0.03) 100%);
            margin-bottom: 14px;
            text-align: center;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
          }
          .big-title { font-size: 1.9rem; font-weight: 700; margin: 0; line-height: 1.2; text-align:center; }
          .subtle { color: var(--text-color-secondary, #6b7280); margin-top: 6px; text-align:center; }
          .card {
            padding: 14px 16px; border-radius: 12px; background: rgba(255,255,255,0.7);
            border: 1px solid rgba(0,0,0,0.06);
          }
          .kpi {
            padding: 12px 14px; border-radius: 12px; color: #fff;
            background: linear-gradient(135deg, #1976D2, #115293);
          }
          .muted { color: #6b7280; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- Utilities ----------
def _to_date(val):
    if val is None or (isinstance(val, str) and val.strip() in {"", "-"}) or pd.isna(val):
        return "-"
    try:
        if isinstance(val, (int, float)) and not math.isnan(val):
            base = datetime(1899, 12, 30)  # Excel serial origin
            return (base + timedelta(days=float(val))).date().isoformat()
    except Exception:
        pass
    if isinstance(val, (pd.Timestamp, datetime)):
        return pd.to_datetime(val).date().isoformat()
    dt = pd.to_datetime(str(val), errors="coerce", dayfirst=True)
    return dt.date().isoformat() if not pd.isna(dt) else str(val)

def _to_pct(val):
    if val is None or (isinstance(val, str) and val.strip() == "") or pd.isna(val):
        return None
    if isinstance(val, str) and val.strip().endswith("%"):
        try:
            return float(val.strip().replace("%", "")) / 100.0
        except Exception:
            return None
    try:
        v = float(val)
        return v / 100.0 if v > 1 else v
    except Exception:
        return None

def _to_num(val):
    if val is None or (isinstance(val, str) and val.strip() in {"", "-", "—"}):
        return np.nan
    try:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        return float(str(val).replace(",", "").strip())
    except Exception:
        return np.nan

def _is_taken(value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s not in {"", "-", "—", "na", "n/a", "not rated", "no", "none", "null", "nan", "nr", "not applicable", "n.a.", "nil"}

def _is_yes(value):
    if value is None or pd.isna(value):
        return False
    return re.search(r"\b(yes|y|true|approved|taken)\b|^1$|✓", str(value).strip(), re.I) is not None

def _is_aaa(value):
    if value is None or pd.isna(value):
        return False
    return AAA_PAT.search(str(value).strip()) is not None

def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _find_col(columns, aliases=None, must_tokens=None, exclude_tokens=None):
    aliases = aliases or []
    must_tokens = [t.replace(" ", "") for t in (must_tokens or [])]
    exclude_tokens = [t.replace(" ", "") for t in (exclude_tokens or [])]
    norm_map = {c: _norm(c) for c in columns}
    norm_aliases = {_norm(a): a for a in aliases}
    for c, n in norm_map.items():
        if n in norm_aliases:
            return c
    for c, n in norm_map.items():
        if all(t in n for t in must_tokens or []) and not any(x in n for x in exclude_tokens or []):
            return c
    return None

def _url(val):
    if not _is_taken(val):
        return None
    s = str(val).strip()
    return s if s.startswith(("http://", "https://")) else f"https://{s}"

def _share_to_csv_url(url: str) -> str:
    if not url:
        return ""
    if "output=csv" in url or "/export" in url:
        return url
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        return url
    sheet_id = m.group(1)
    gid = "0"
    if "#gid=" in url:
        gid = url.split("#gid=")[-1].split("&")[0]
    else:
        qs_gid = parse_qs(urlparse(url).query).get("gid", [None])[0]
        if qs_gid:
            gid = str(qs_gid)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def _num_series(df: pd.DataFrame, colname: str, fill=np.nan) -> pd.Series:
    if colname and colname in df.columns:
        return pd.to_numeric(df[colname].map(_to_num), errors="coerce")
    return pd.Series([fill] * len(df), index=df.index, dtype="float64")

def _standardize_selector_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    entity_col = _find_col(
        cols,
        aliases=["Entity", "Entity Name", "REIT", "REIT Name", "Name of REIT",
                 "Trust", "Trust Name", "Issuer", "Issuer Name", "InvIT / REIT Name", "InvIT Name", "Name of InvIT"]
    )
    fy_col = _find_col(
        cols,
        aliases=["Financial Year", "Fin Year", "FY", "Financial Yr", "Year"],
        must_tokens=["financial", "year"], exclude_tokens=["quarter", "qtr"]
    ) or _find_col(cols, aliases=[], must_tokens=["year"], exclude_tokens=["quarter", "qtr"])
    qtr_col = _find_col(
        cols,
        aliases=["Quarter Ended", "Quarter", "Qtr", "Q/E", "Quarter (Ended)"],
        must_tokens=["quarter"], exclude_tokens=["year"]
    )
    if entity_col:
        df = df.rename(columns={entity_col: ENT_COL})
    else:
        df[ENT_COL] = np.nan
    if fy_col:
        df = df.rename(columns={fy_col: FY_COL})
    else:
        df[FY_COL] = np.nan
    if qtr_col:
        df = df.rename(columns={qtr_col: QTR_COL})
    else:
        if QTR_COL not in df.columns:
            df[QTR_COL] = np.nan
    df.attrs["__selector_map__"] = {"Entity": entity_col, "Financial Year": fy_col, "Quarter Ended": qtr_col}
    return df

def _quarter_sort(values):
    order = {"June": 0, "Sept": 1, "Sep": 1, "December": 2, "Dec": 2, "Mar": 3, "March": 3}
    return sorted(values, key=lambda v: order.get(str(v), 99))

# ---------- Universal URL Loader ----------
@st.cache_data(show_spinner=False, ttl=300)
def load_table_url(url: str) -> pd.DataFrame:
    if not url or not str(url).strip():
        raise ValueError("Empty URL.")
    url = url.strip()
    if "docs.google.com/spreadsheets" in url:
        url = _share_to_csv_url(url)
    # CSV
    try:
        df = pd.read_csv(url)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    # Excel (bytes)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=25)
        resp.raise_for_status()
        try:
            df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass
        # JSON in same response
        try:
            data = resp.json()
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                records = None
                for key in ["data", "rows", "items", "records", "result"]:
                    if key in data and isinstance(data[key], list):
                        records = data[key]; break
                if records is None:
                    records = [data]
                df = pd.json_normalize(records)
            else:
                df = pd.DataFrame(data)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass
    except Exception:
        pass
    # JSON via pandas
    try:
        df = pd.read_json(url)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    # HTML table(s)
    try:
        tables = pd.read_html(url)
        if tables:
            df = max(tables, key=lambda t: (t.shape[0] * t.shape[1]))
            return df
    except Exception:
        pass
    raise ValueError("Couldn't parse the URL as CSV, Excel, JSON, or an HTML table. Ensure it is publicly accessible.")
