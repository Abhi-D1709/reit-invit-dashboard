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
DEFAULT_INVIT_DIR_URL  = "https://docs.google.com/spreadsheets/d/1twj3iCRDOk46Hb8xGHgkDCW5hfUp170EltXKx0aepm8/edit?usp=sharing"

# Trading entities (REIT/InvIT mapping) - defined in utils/constants.py so jobs/ can use it without streamlit
from utils.constants import ENTITIES_SHEET_ID, ENTITIES_SHEET_CSV  # noqa: E402,F401

# Sponsor Holding (REIT)
DEFAULT_REIT_SPON_URL  = "https://docs.google.com/spreadsheets/d/1PnuNGHDskqBZt4WUO8JpmssaQ3nOUvLJolfwB7T5zDE/export?format=csv&gid=1466135872"

# NDCF (REITs)
NDCF_REITS_SHEET_URL = "https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing"
NDCF_INVITS_SHEET_URL = ""

# Sponsor Holding (InvIT)
DEFAULT_INVIT_SPON_URL = ""

# Governance Data (REITs)
GOVERNANCE_REIT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ETx5UZKQQyZKxkF4fFJ4R9wa7i7TNp7EXIhHWiVYG7s/edit?usp=sharing"

# Valuation Data (REITs)
VALUATION_REIT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1dujQ1XpnorGgXvmrlSjWuBkQFGifnC39sqbG98W9TrQ/edit?usp=sharing"

# Related Party Transactions (REITs)
RPT_REIT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1zn1XtA1DHF0VANTkowFZqenUUDSISlz1LFscT8SI-yM/edit?usp=sharing"

# Investment Conditions (REITs)
INVESTMENT_REIT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1OqnSFP_PkzsoIr4BaP66-5XqdNlGVTbK9DR3thqp9JA/edit?usp=sharing"

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

# Placeholders people type instead of leaving a cell empty. All of them mean "no value", never zero.
_MISSING_TEXT = {
    "", "-", "--", "\u2014", "\u2013", "na", "n/a", "n.a.", "n.a", "nil", "none", "null", "nan",
    "not applicable", "not available", "#n/a", "#div/0!", "#value!", "#ref!", "#name?",
}


def parse_number(val) -> float:
    """Number from a sheet cell; NaN when the cell is blank, a placeholder ("-", "NA", "#DIV/0!")
    or cannot be read. Never returns 0 for "no value": a missing figure must stay missing so
    checks can say "insufficient data" instead of passing or failing on an invented zero.

    Handles thousands separators (also Indian 3,16,124), currency marks, (1,200) as -1200 and a
    unicode minus. A trailing % is ignored (the number is returned as written); use
    resolve_percent_units for percentages, where the unit matters.
    """
    if val is None or isinstance(val, bool):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val) if math.isfinite(float(val)) else np.nan
    text = str(val).replace("\u00a0", " ").strip()
    if text.lower() in _MISSING_TEXT:
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace("\u2212", "-")
    for junk in (",", "\u20b9", "$", " ", "%"):
        text = text.replace(junk, "")
    for prefix in ("rs.", "rs", "inr"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    try:
        number = float(text)
    except ValueError:
        return np.nan
    if not math.isfinite(number):
        return np.nan
    return -abs(number) if negative else number


_to_num = parse_number  # historical name


def resolve_percent_units(raw: pd.Series, reference: pd.Series | None = None, groups: pd.Series | None = None):
    """Turn a column of percentages typed in mixed conventions into fractions (0.25 = 25%).

    People type the same ratio as "26%", "26.09" (percent points) or "0.2609" (fraction), and a
    single column can mix them, even per entity. Guessing per cell is wrong (0.85 can be 85% or
    0.85%), so each cell is decided by the best evidence available, in this order:

      1. it has a % sign                          -> percent points
      2. |value| > 1.5                             -> percent points (a fraction that large is implausible)
      3. `reference` (the same ratio computed from other columns) is known
                                                   -> whichever reading lands closer to it
      4. the same group (entity) has other rows that are clearly percent points
                                                   -> percent points
      5. otherwise                                 -> fraction

    Returns (fractions, how): `how` says which rule decided each cell ("" when blank).
    """
    cells = raw.astype("object")
    has_sign = cells.map(lambda v: isinstance(v, str) and v.strip().endswith("%"))
    number = cells.map(parse_number)
    fractions = pd.Series(np.nan, index=cells.index, dtype="float64")
    how = pd.Series("", index=cells.index, dtype="object")

    fractions[has_sign] = number[has_sign] / 100.0
    how[has_sign] = "with % sign"

    bare = ~has_sign & number.notna()
    big = bare & (number.abs() > 1.5)
    fractions[big] = number[big] / 100.0
    how[big] = "percent points (value above 1.5)"

    open_ = bare & ~big
    if reference is not None and open_.any():
        ref = pd.to_numeric(reference, errors="coerce")
        usable = open_ & ref.notna()
        as_points = (number / 100.0 - ref).abs()
        as_fraction = (number - ref).abs()
        points = usable & (as_points < as_fraction)
        fractions[points] = number[points] / 100.0
        how[points] = "closest to the value computed from the components"
        fraction = usable & ~points
        fractions[fraction] = number[fraction]
        how[fraction] = "closest to the value computed from the components"
        open_ = open_ & ~usable

    if groups is not None and open_.any():
        points_groups = set(groups[big].dropna())
        points = open_ & groups.isin(points_groups)
        fractions[points] = number[points] / 100.0
        how[points] = "same entity's other rows are percent points"
        open_ = open_ & ~points

    fractions[open_] = number[open_]
    how[open_] = "assumed a fraction (value between -1.5 and 1.5)"
    return fractions, how


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
    if not must_tokens:
        return None  # no alias matched and no tokens to search by: report "not found", never guess a column
    for c, n in norm_map.items():
        if all(t in n for t in must_tokens) and not any(x in n for x in exclude_tokens):
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
        return pd.to_numeric(df[colname].map(parse_number), errors="coerce")
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