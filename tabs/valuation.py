from __future__ import annotations

import re
import math
import time
import html
import typing as t
from datetime import date, datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup  # type: ignore
import streamlit as st

# --------------------------------------------------------------------
# Config: read the Google Sheet URL from utils.common (no hardcoding)
# --------------------------------------------------------------------
try:
    from utils.common import VALUATION_REIT_SHEET_URL  # type: ignore
    _SHEET_URL = str(VALUATION_REIT_SHEET_URL).strip()
except Exception:
    _SHEET_URL = ""

if not _SHEET_URL:
    st.warning(
        "VALUATION_REIT_SHEET_URL was not found in utils/common.py. "
        "Please add it there as a public (view) Google Sheet URL."
    )

# --------------------------------------------------------------------
# Typing helpers (use typing, not runtime variables, in type hints)
# --------------------------------------------------------------------
Row = t.Dict[str, t.Any]
Rows = t.List[Row]
DF = pd.DataFrame

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def _doc_id_from_share_url(url: str) -> str | None:
    """
    Extract the Google Sheet document ID from a standard sharing URL.
    """
    m = re.search(r"/d/([^/]+)/", url)
    return m.group(1) if m else None


def _csv_export_url(share_url: str, sheet_name: str) -> str:
    """
    Build the 'gviz' CSV export URL for a given sheet name.

    We purposely use the 'sheet' param (by title) rather than gid. This
    avoids hardcoding gids and works well as long as the sheet name is stable.
    """
    doc_id = _doc_id_from_share_url(share_url or "")
    if not doc_id:
        return ""
    # Google sheets CSV export by sheet title
    return f"https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv&sheet={requests.utils.quote(sheet_name)}"


def _parse_date(value: t.Any) -> pd.Timestamp | pd.NaTType:
    """
    Parse dates that may come as strings like '14/09/2020', 'NA', '', or actual datetimes.
    We assume day-first when ambiguous.
    """
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.to_datetime(value)
    if value is None:
        return pd.NaT
    s = str(value).strip()
    if not s or s.upper() == "NA" or s == "-":
        return pd.NaT
    # Unescape HTML and normalize whitespace
    s = re.sub(r"\s+", " ", html.unescape(s))
    try:
        # dayfirst covers dd/mm/yyyy formats
        ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
    except Exception:
        ts = pd.NaT
    return ts


def _years_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    """
    Approximate year difference; good enough for 4-year compliance.
    """
    return float((d2 - d1).days) / 365.25


def _normalize_name(x: str) -> str:
    """
    Lowercase + strip honorifics + compress spaces + remove punctuation
    for loose matching (fallback when Reg. No. is missing).
    """
    s = (x or "").lower()
    # remove common honorifics
    s = re.sub(r"\b(mr|mrs|ms|dr|shri|smt|kum)\.?\s+", "", s)
    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------------------------------------------------
# Data loaders (cache heavily)
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_valuers_sheet(sheet_url: str) -> DF:
    """
    Reads Sheet1 of your valuation workbook:
    Required columns:
      - 'Name of REIT'
      - 'Finanical Year' (note the original misspelling, we accept both)
      - 'Financial Year' (also accepted)
      - 'Name of Valuer'
      - 'Date of Appointmnet' (original header spelling)
      - 'Date of Resignation'
      - 'IBBI Registration No.'
    """
    csv_url = _csv_export_url(sheet_url, "Sheet1")
    if not csv_url:
        return pd.DataFrame()
    df = pd.read_csv(csv_url)

    # Normalize headers
    df.columns = [c.strip() for c in df.columns]

    # Handle both "Finanical Year" (as in sheet) and "Financial Year"
    if "Financial Year" in df.columns and "Finanical Year" not in df.columns:
        df.rename(columns={"Financial Year": "Finanical Year"}, inplace=True)

    # Ensure expected columns are present (silently create empty if missing)
    expected = [
        "Name of REIT",
        "Finanical Year",
        "Name of Valuer",
        "Date of Appointmnet",
        "Date of Resignation",
        "IBBI Registration No.",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA

    # Parse date columns
    df["Date of Appointmnet"] = df["Date of Appointmnet"].apply(_parse_date)
    df["Date of Resignation"] = df["Date of Resignation"].apply(_parse_date)

    return df


def _extract_table_rows(tbl: BeautifulSoup) -> Rows:
    """
    Convert a BeautifulSoup <table> into list-of-dicts by header.
    Safe typing: returns List[Dict[str, Any]].
    """
    headers: t.List[str] = []
    rows: Rows = []

    thead = tbl.find("thead")
    if thead:
        ths = thead.find_all("th")
        headers = [th.get_text(strip=True) for th in ths]

    tbody = tbl.find("tbody")
    if not tbody:
        return rows

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        values = [td.get_text(strip=True) for td in tds]
        # Align to headers; if headers short, fall back to positional dict
        if headers and len(values) == len(headers):
            row = {headers[i]: values[i] for i in range(len(headers))}
        else:
            row = {f"col_{i}": values[i] for i in range(len(values))}
        rows.append(row)
    return rows


@st.cache_data(show_spinner=False)
def scrape_ibbi_registry() -> DF:
    """
    Scrape both IBBI individual RVs and RVO-entities.
    Returns a combined DataFrame with columns:
        'source' = {'individual','entity'}
        'Registration No.'
        'Name'
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; REIT-INVIT-Dashboard/1.0)",
        }
    )

    # Individuals
    ind_url = "https://ibbi.gov.in/service-provider/rvs?page={page}"
    ent_url = "https://ibbi.gov.in/service-provider/rvo-entities?page={page}"

    all_rows: Rows = []

    def crawl(base_url: str, kind: str) -> None:
        page = 1
        while True:
            url = base_url.format(page=page)
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            tables = soup.find_all("table", class_="reporttable")
            if not tables:
                # the entities table uses a slightly different class set; be generous
                tables = soup.find_all("table")
            found_any = False
            for tbl in tables:
                rows = _extract_table_rows(tbl)
                if not rows:
                    continue
                found_any = True
                # Heuristic to find name & reg no columns by header names
                for r in rows:
                    keys = {k.lower(): k for k in r.keys()}
                    reg_key = (
                        keys.get("registration no.")
                        or keys.get("registration no")
                        or keys.get("reg. no.")
                        or keys.get("reg no.")
                    )
                    # Individuals: "Name of RV"; Entities: "Name of RVE"
                    name_key = (
                        keys.get("name of rv")
                        or keys.get("name of rve")
                        or keys.get("name")
                    )
                    if not reg_key or not name_key:
                        continue
                    all_rows.append(
                        {
                            "source": kind,
                            "Registration No.": r.get(reg_key, "").strip(),
                            "Name": r.get(name_key, "").strip(),
                        }
                    )
            if not found_any:
                break
            page += 1
            # be a good citizen
            time.sleep(0.4)

    crawl(ind_url, "individual")
    crawl(ent_url, "entity")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["norm_name"] = df["Name"].map(_normalize_name)
    df["reg_norm"] = df["Registration No."].str.replace(r"\s+", "", regex=True).str.upper()
    df.drop_duplicates(subset=["reg_norm", "norm_name"], inplace=True)
    return df


# --------------------------------------------------------------------
# Business rules
# --------------------------------------------------------------------
def check_tenure_leq_4yrs(appoint: pd.Timestamp, resign: pd.Timestamp | pd.NaTType) -> tuple[bool, float | None]:
    """
    Returns (ok, years). If resignation is missing, compares to today.
    """
    if pd.isna(appoint):
        return False, None
    end_dt = resign if (isinstance(resign, pd.Timestamp) and not pd.isna(resign)) else pd.Timestamp(date.today())
    years = _years_between(appoint, end_dt)
    return years <= 4.0, years


def check_ibbi_registered(name: str, reg_no: str, registry: DF) -> bool:
    """
    Returns True if the valuer is found in the IBBI registry either by exact
    registration number match (preferred) or by normalized name fallback.
    """
    if registry is None or registry.empty:
        return False

    reg_norm = (reg_no or "").replace(" ", "").upper()
    if reg_norm:
        hit = registry["reg_norm"] == reg_norm
        if bool(hit.any()):
            return True

    nm = _normalize_name(name or "")
    if not nm:
        return False
    hit = registry["norm_name"] == nm
    return bool(hit.any())


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
def render_valuation() -> None:
    st.markdown("### Valuation — Valuer Compliance Checks")

    # Load data
    val_df = load_valuers_sheet(_SHEET_URL)
    if val_df.empty:
        st.info("No valuation records found.")
        return

    # Selections
    reits = sorted([x for x in val_df["Name of REIT"].dropna().unique().tolist() if str(x).strip()])
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_reit = st.selectbox("Choose REIT", reits, index=0 if reits else None)
    with col2:
        fy_opts = (
            val_df.loc[val_df["Name of REIT"] == selected_reit, "Finanical Year"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        fy_opts = sorted(fy_opts)
        selected_fy = st.selectbox("Financial Year", fy_opts, index=0 if fy_opts else None)

    filtered = val_df[(val_df["Name of REIT"] == selected_reit) & (val_df["Finanical Year"].astype(str) == str(selected_fy))]
    if filtered.empty:
        st.warning("No rows for the chosen REIT and year.")
        return

    # Pull IBBI registry once (cached)
    with st.spinner("Refreshing IBBI registry…"):
        ibbi = scrape_ibbi_registry()

    # Compute results
    out_rows: Rows = []
    for _, r in filtered.iterrows():
        name = str(r.get("Name of Valuer", "")).strip()
        regno = str(r.get("IBBI Registration No.", "")).strip()
        appoint = t.cast(pd.Timestamp, r.get("Date of Appointmnet"))
        resign = t.cast(pd.Timestamp, r.get("Date of Resignation"))

        ok_tenure, yrs = check_tenure_leq_4yrs(appoint, resign)
        ok_reg = check_ibbi_registered(name, regno, ibbi)

        out_rows.append(
            {
                "Name of REIT": r.get("Name of REIT"),
                "Financial Year": r.get("Finanical Year"),
                "Valuer": name,
                "IBBI Reg No.": regno,
                "Appointment": appoint.date().isoformat() if isinstance(appoint, pd.Timestamp) and not pd.isna(appoint) else "",
                "Resignation": resign.date().isoformat() if isinstance(resign, pd.Timestamp) and not pd.isna(resign) else "",
                "Tenure ≤ 4 years": "✅" if ok_tenure else "❌",
                "Tenure (yrs)": f"{yrs:.2f}" if yrs is not None else "",
                "Registered with IBBI": "✅" if ok_reg else "❌",
            }
        )

    res_df = pd.DataFrame(out_rows)

    # Small summary
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows analysed", len(res_df))
    with c2:
        st.metric("Registered with IBBI", int((res_df["Registered with IBBI"] == "✅").sum()))

    # Table
    st.dataframe(
        res_df,
        use_container_width=True,
        hide_index=True,
    )

# For backward-compatibility with your page wrapper
def render():
    render_valuation()
