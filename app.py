# app.py
import io
import re
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

# ======================== Page Config & Styling ========================
st.set_page_config(page_title="REIT / InvIT Dashboard", page_icon="📊", layout="wide")

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

# ======================== Defaults ========================
DEFAULT_REIT_BORR_URL  = "https://docs.google.com/spreadsheets/d/1OugwmVbR2BXjWcRGOlLhqrg3APVv9R17LYpZPDeFDkw/edit?usp=sharing"
DEFAULT_INVIT_BORR_URL = "https://docs.google.com/spreadsheets/d/1Zqi5VWeS2GSfhWa0gVPruIHdYzYM6luWalOv_8mhHsc/edit?usp=sharing"

# Fund Raising Google Sheets you shared:
DEFAULT_REIT_FUND_URL  = "https://docs.google.com/spreadsheets/d/1cuH2odCdJpnP5E0trvroQWzB4rXFgWGpjhHcAuX81Hs/edit?usp=sharing"
DEFAULT_INVIT_FUND_URL = "https://docs.google.com/spreadsheets/d/1eepPHnjo31G3ueeQTGxmVT7iY9cXh3NRgpDJuq6ygS8/edit?usp=sharing"

# Canonical internal names for selectors
ENT_COL = "__Entity__"
FY_COL  = "__FinancialYear__"
QTR_COL = "__QuarterEnded__"    # used by Borrowings

EPS = 1e-9
AAA_PAT = re.compile(r'(^|\W)(AAA|Aaa)($|\W)', re.I)

# ======================== Helpers (shared) ========================
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
    """Flexible finder: exact alias (case/space-insensitive) or token match."""
    aliases = aliases or []
    must_tokens = [t.replace(" ", "") for t in (must_tokens or [])]
    exclude_tokens = [t.replace(" ", "") for t in (exclude_tokens or [])]
    norm_map = {c: _norm(c) for c in columns}
    norm_aliases = {_norm(a): a for a in aliases}
    # 1) exact alias
    for c, n in norm_map.items():
        if n in norm_aliases:
            return c
    # 2) token-based
    for c, n in norm_map.items():
        if all(t in n for t in must_tokens) and not any(x in n for x in exclude_tokens):
            return c
    return None

def _url(val):
    if not _is_taken(val):
        return None
    s = str(val).strip()
    return s if s.startswith(("http://", "https://")) else f"https://{s}"

def _quarter_sort(values):
    order = {"June": 0, "Sept": 1, "Sep": 1, "December": 2, "Dec": 2, "Mar": 3, "March": 3}
    return sorted(values, key=lambda v: order.get(str(v), 99))

def _share_to_csv_url(url: str) -> str:
    """Google Sheets view URL -> export CSV URL (keeps gid if present)."""
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

def _to_num(val):
    """Parse numbers from strings with commas/Indian formatting; '-'->NaN."""
    if val is None or (isinstance(val, str) and val.strip() in {"", "-", "—"}):
        return np.nan
    try:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        s = str(val).strip().replace(",", "")
        return float(s)
    except Exception:
        return np.nan

def _num_series(df: pd.DataFrame, colname: str, fill=np.nan) -> pd.Series:
    if colname and colname in df.columns:
        return pd.to_numeric(df[colname].map(_to_num), errors="coerce")
    return pd.Series([fill] * len(df), index=df.index, dtype="float64")

# ======================== Universal URL Loader ========================
@st.cache_data(show_spinner=False, ttl=300)
def load_table_url(url: str) -> pd.DataFrame:
    """
    Accepts public URLs for:
      - Google Sheets (view link auto-converted to CSV export)
      - CSV
      - XLSX
      - JSON
      - HTML pages containing <table>
    Returns a raw dataframe (no schema normalization).
    """
    if not url or not str(url).strip():
        raise ValueError("Empty URL.")
    url = url.strip()

    # Google Sheets view link → CSV export
    if "docs.google.com/spreadsheets" in url:
        url = _share_to_csv_url(url)

    # 1) Try CSV directly
    try:
        df = pd.read_csv(url)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass

    # 2) Try Excel via bytes
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=25)
        resp.raise_for_status()
        # Try as Excel
        try:
            df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass
        # 3) Try JSON from same response
        try:
            data = resp.json()
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                records = None
                for key in ["data", "rows", "items", "records", "result"]:
                    if key in data and isinstance(data[key], list):
                        records = data[key]
                        break
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

    # 4) Try JSON via pandas (URL)
    try:
        df = pd.read_json(url)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass

    # 5) Try HTML tables
    try:
        tables = pd.read_html(url)
        if tables:
            df = max(tables, key=lambda t: (t.shape[0] * t.shape[1]))
            return df
    except Exception:
        pass

    raise ValueError("Couldn't parse the URL as CSV, Excel, JSON, or an HTML table. Ensure it is publicly accessible.")

# ======================== Borrowings Processing ========================
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
    # Missing is okay for Fund Raising (no Quarter); we still add placeholders
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

def _process_borrowings_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = _standardize_selector_columns(df)

    cols = df.columns
    borrow_col = _find_col(cols, aliases=["Borrowings","A. Borrowings","A - Borrowings","A Borrowings"], must_tokens=["borrow"])
    defer_col  = _find_col(cols, aliases=["Deferred Payments","B. Deferred Payments","Deferred Payment"], must_tokens=["defer","payment"])
    cash_col   = _find_col(cols, aliases=["Cash and Cash Equivalents","C. Cash and Cash Equivalents","Cash & Cash Equivalents","Cash and cash equivalents"], must_tokens=["cash","equivalent"])
    assets_col = _find_col(cols, aliases=["Value of REIT Assets","D. Value of REIT Assets","Value of Assets","Value of InvIT Assets","Value of Trust Assets"], must_tokens=["value","asset"])

    A = _num_series(df, borrow_col, 0.0)
    B = _num_series(df, defer_col, 0.0)
    C = _num_series(df, cash_col, 0.0)
    D = _num_series(df, assets_col, np.nan)

    nbr_col = _find_col(cols, aliases=["Net Borrowings Ratio (NBR)"], must_tokens=["borrow","ratio","nbr"])
    if nbr_col:
        df["NBR_ratio"] = df[nbr_col].apply(_to_pct)

    if "NBR_ratio" not in df.columns or df["NBR_ratio"].isna().any():
        D_safe = D.replace(0, np.nan)
        computed = (A.add(B, fill_value=0).sub(C, fill_value=0)) / D_safe
        df["NBR_ratio"] = df.get("NBR_ratio", computed).fillna(computed)

    for col in [
        "Date of Publishing Credit Rating CRA1",
        "Date of Publishing Credit Rating CRA2",
        "Date of meeting for Unitholder Approval",
        "Date Of intimation to Trustee",
    ]:
        if col in df.columns:
            df[f"{col} (fmt)"] = df[col].apply(_to_date)

    df.attrs["__matched_cols__"] = {
        "Borrowings": borrow_col, "Deferred Payments": defer_col,
        "Cash and Cash Equivalents": cash_col, "Value of REIT/Trust Assets": assets_col,
        "NBR source": nbr_col or "computed",
    }
    return df

@st.cache_data(show_spinner=False, ttl=300)
def load_borrowings_url(url: str) -> pd.DataFrame:
    df = load_table_url(url)
    return _process_borrowings_df(df)

# ======================== Fund Raising Processing ========================
def _process_fundraising_df(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """
    Normalize to common schema:
    - __Entity__, __FinancialYear__
    - Listed on, Public/ Private Listed
    - Date of Fund raising (fmt)
    - Type of Issue, Category of Fund Raising
    - Amount of Fund Raised (float), No. of Units Issued (float), Unit Capital... (float)
    """
    df.columns = [c.strip() for c in df.columns]
    df = _standardize_selector_columns(df)

    cols = df.columns
    # Column discovery
    listed_on_col   = _find_col(cols, aliases=["Listed on"])
    pp_listed_col   = _find_col(cols, aliases=["Public/ Private Listed"])
    date_col        = _find_col(cols, aliases=["Date of Fund raising", "Date of Fund Raising"])
    type_col        = _find_col(cols, aliases=["Type of Issue"])
    cat_col         = _find_col(cols, aliases=["Category of Fund Raising"])
    amt_col         = _find_col(cols, aliases=["Amount of Fund Raised", "Amount Raised"])
    units_col       = _find_col(cols, aliases=["No. of Units Issued"])
    unitcap_col     = _find_col(cols, aliases=["Unit Capital at the end of Fund Raising", "Unit Capital at End"])

    # Numeric normalizations
    df["Amount of Fund Raised (num)"] = _num_series(df, amt_col)
    df["No. of Units Issued (num)"] = _num_series(df, units_col)
    df["Unit Capital at End (num)"] = _num_series(df, unitcap_col)

    # Date normalization
    if date_col:
        df["Date of Fund raising (fmt)"] = df[date_col].apply(_to_date)
        df["__date__"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    else:
        df["Date of Fund raising (fmt)"] = "-"
        df["__date__"] = pd.NaT

    # Save for UI
    df.attrs["__fund_cols__"] = {
        "listed_on": listed_on_col,
        "public_private": pp_listed_col,
        "date": date_col,
        "type": type_col,
        "category": cat_col,
        "amount_num": "Amount of Fund Raised (num)",
        "units_num": "No. of Units Issued (num)",
        "unitcap_num": "Unit Capital at End (num)",
    }
    return df

@st.cache_data(show_spinner=False, ttl=300)
def load_fundraising_url(url: str, segment: str) -> pd.DataFrame:
    df = load_table_url(url)
    return _process_fundraising_df(df, segment=segment)

# ======================== Borrowings UI (with InvIT rules) ========================
def render_borrowings_panel(default_url: str, segment_label: str, ruleset: str = "reit"):
    st.subheader("Data Source")
    data_url = st.text_input("Paste a public URL (Google Sheet / CSV / XLSX / JSON / HTML table):", value=default_url, key=f"data_url_{segment_label}")

    if not data_url.strip():
        st.warning("Please provide a data URL.")
        st.stop()

    try:
        df = load_borrowings_url(data_url.strip())
    except Exception as e:
        st.error(f"Could not read the URL. Make sure it’s publicly accessible.\n\nDetails: {e}")
        st.stop()

    # Filters
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            entities = sorted(df[ENT_COL].dropna().astype(str).unique())
            entity = st.selectbox("Entity", entities, key=f"entity_{segment_label}")
        with c2:
            fy_options = sorted(df.loc[df[ENT_COL] == entity, FY_COL].dropna().astype(str).unique())
            fy = st.selectbox("Financial Year", fy_options, key=f"fy_{segment_label}")
        with c3:
            qtr_present = df.loc[(df[ENT_COL] == entity) & (df[FY_COL] == fy), QTR_COL].dropna().astype(str).unique().tolist()
            qtr = st.selectbox("Quarter", _quarter_sort(qtr_present), key=f"qtr_{segment_label}")

    row_df = df[(df[ENT_COL] == entity) & (df[FY_COL] == fy) & (df[QTR_COL] == qtr)]
    if row_df.empty:
        st.warning("No data found for the selected filters.")
        st.stop()
    row = row_df.iloc[0]

    st.markdown("### Overview")

    # KPI + Breakup grid
    colA, colB = st.columns([0.9, 1.1])

    with colA:
        nbr = row.get("NBR_ratio", None)
        nbr_display = "-" if nbr is None or pd.isna(nbr) else f"{float(nbr)*100:.2f}%"
        st.markdown(
            '<div class="kpi">📊 <b>Net Borrowings Ratio</b><br>'
            f'<span style="font-size:28px;font-weight:700;">{nbr_display}</span></div>',
            unsafe_allow_html=True
        )
        if isinstance(nbr, (int, float)) and not pd.isna(nbr):
            st.progress(min(max(float(nbr), 0.0), 1.0))

    with colB:
        m = df.attrs.get("__matched_cols__", {})
        a_label = m.get("Borrowings") or "Borrowings"
        b_label = m.get("Deferred Payments") or "Deferred Payments"
        c_label = m.get("Cash and Cash Equivalents") or "Cash and Cash Equivalents"
        d_label = m.get("Value of REIT/Trust Assets") or "Value of REIT Assets"
        st.markdown("**Breakup**")
        st.write(
            f"""
- **A. Borrowings**: {row.get(a_label, "-")}
- **B. Deferred Payments**: {row.get(b_label, "-")}
- **C. Cash and Cash Equivalents**: {row.get(c_label, "-")}
- **D. Value of REIT Assets**: {row.get(d_label, "-")}
"""
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- RULES & VISIBILITY --------------------
    if ruleset == "invit":
        # 70% hard cap alert
        if isinstance(nbr, (int, float)) and not pd.isna(nbr) and (nbr > 0.70 + EPS):
            st.error(f"ALERT: NBR is {float(nbr)*100:.2f}% which exceeds the 70% cap for InvITs.")
        # For InvIT, show details only when NBR > 25%
        show_sections = isinstance(nbr, (int, float)) and not pd.isna(nbr) and (nbr > 0.25 + EPS)
    else:
        # REIT: show when NBR ≥ 25%
        show_sections = isinstance(nbr, (int, float)) and not pd.isna(nbr) and (nbr >= 0.25 - EPS)

    if not show_sections:
        st.info("NBR is below the threshold. Credit Rating, Unitholder Approval, and Additional Compliances are not required to be displayed.")
        return

    # -------------------- Common extraction --------------------
    cols = row.index
    # CRA1 / CRA2
    cra1_rating_col = _find_col(cols, aliases=["Credit Rating CRA1"])
    cra2_rating_col = _find_col(cols, aliases=["Credit Rating CRA2"])
    cra1_name_col   = _find_col(cols, aliases=["Name of CRA1"])
    cra2_name_col   = _find_col(cols, aliases=["Name of CRA2"])
    cra1_date_col   = _find_col(cols, aliases=["Date of Publishing Credit Rating CRA1 (fmt)", "Date of Publishing Credit Rating CRA1"])
    cra2_date_col   = _find_col(cols, aliases=["Date of Publishing Credit Rating CRA2 (fmt)", "Date of Publishing Credit Rating CRA2"])
    cra1_link_col   = _find_col(cols, aliases=["Weblink of CRA1 Disclosure (CRA/Exchange)"])
    cra2_link_col   = _find_col(cols, aliases=["Weblink of CRA2 Disclosure (CRA/Exchange)"])
    cra1_rating = row.get(cra1_rating_col); cra2_rating = row.get(cra2_rating_col)
    cra1_name   = row.get(cra1_name_col);   cra2_name   = row.get(cra2_name_col)
    cra1_date   = row.get(cra1_date_col);   cra2_date   = row.get(cra2_date_col)
    cra1_link   = _url(row.get(cra1_link_col)); cra2_link = _url(row.get(cra2_link_col))
    credit_taken_any = _is_taken(cra1_rating) or _is_taken(cra2_rating)
    aaa_ok = _is_aaa(cra1_rating) or _is_aaa(cra2_rating)

    # Unitholder Approval
    ua_col = _find_col(
        cols,
        aliases=["Unitholder Approval", "Unitholder approval"],
        must_tokens=["unitholder", "approval"],
        exclude_tokens=["date", "meeting", "weblink", "notice", "votes", "record", "favour", "against", "total"]
    )
    unitholder_approval_val = row.get(ua_col)
    unit_taken = _is_yes(unitholder_approval_val)

    # Alerts per rules
    missing = []
    if ruleset == "invit":
        if isinstance(nbr, (int, float)) and not pd.isna(nbr):
            if nbr > 0.49 + EPS:
                if not aaa_ok:     missing.append("AAA Credit Rating")
                if not unit_taken: missing.append("Unitholder Approval")
            elif nbr > 0.25 + EPS:
                if not credit_taken_any: missing.append("Credit Rating")
                if not unit_taken:       missing.append("Unitholder Approval")
    else:
        if not credit_taken_any: missing.append("Credit Rating")
        if not unit_taken:       missing.append("Unitholder Approval")

    if missing:
        if len(missing) == 2:
            msg = f"Both {missing[0]} and {missing[1]} are not taken / not available for this period."
        else:
            msg = f"{missing[0]} is not taken / not available for this period."
        st.error(f"ALERT: {msg}")

    # Credit Rating (dual)
    st.markdown("### Credit Rating")
    cr1, cr2 = st.columns(2)
    with cr1:
        st.markdown("**CRA1**")
        st.write(f"**Rating**: {cra1_rating if _is_taken(cra1_rating) else '-'}")
        st.write(f"**Name of CRA**: {cra1_name if _is_taken(cra1_name) else '-'}")
        st.write(f"**Date of Publishing Credit Rating**: {_to_date(cra1_date) if _is_taken(cra1_date) else '-'}")
        if cra1_link: st.markdown(f"[CRA1 Disclosure Link]({cra1_link})")
        st.markdown('</div>', unsafe_allow_html=True)
    with cr2:
        st.markdown("**CRA2**")
        st.write(f"**Rating**: {cra2_rating if _is_taken(cra2_rating) else '-'}")
        st.write(f"**Name of CRA**: {cra2_name if _is_taken(cra2_name) else '-'}")
        st.write(f"**Date of Publishing Credit Rating**: {_to_date(cra2_date) if _is_taken(cra2_date) else '-'}")
        if cra2_link: st.markdown(f"[CRA2 Disclosure Link]({cra2_link})")
        st.markdown('</div>', unsafe_allow_html=True)

    # Optional combined metric
    updown2 = next((row.get(c) for c in [
        "No. of Rating Upgrades/Downgrades CRA1",
        "No. of Rating Upgrades/Downgrades CRA2",
        "No. of Rating Upgrades/Downgrades"
    ] if c in cols), "-")
    st.write(f"**No. of Rating Upgrades/Downgrades**: {updown2}")

    st.markdown("---")

    # Unitholder Approval
    st.markdown("### Unitholder Approval")
    approval_display = "Yes" if _is_yes(unitholder_approval_val) else ("No" if _is_taken(unitholder_approval_val) else "-")
    st.write(f"**Approval Taken**: {approval_display}")
    st.write(f"**Date of meeting for Unitholder Approval**: {row.get('Date of meeting for Unitholder Approval (fmt)', '-')}")
    link_um = _url(next((row.get(c) for c in [
        "Weblink of Disclosure of Outcome of Unitholder Meeting (Exchange)",
        "Weblink of Disclosure of Notice for Unitholder Meeting (Exchange)"
    ] if c in cols), "-"))
    if link_um:
        st.markdown(f"[Disclosure on Exchange]({link_um})")
    st.write(f"**Total No. of Unitholders on record date**: {row.get('Total No. of Unitholders on record date', '-')}")
    st.write(f"**Total No. of Votes Cast**: {row.get('Total No. of Votes Cast', '-')}")
    st.write(f"**Votes Cast (Favour/Against)**: {row.get('Votes Cast in Favour/Votes Cast Against', '-')}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Additional Compliances
    st.markdown("### Additional Compliances")
    st.write(f"**Whether NBR > 25% due to market movement?**  {row.get('Whether NBR>25% on account of market movement?', '-')}")
    st.write(f"**Date of intimation to Trustee**: {row.get('Date Of intimation to Trustee (fmt)', '-')}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        **Common Links**
        - [Indiabondinfo](https://www.indiabondinfo.nsdl.com/)
        - [CDSL BondInfo](https://www.cdslindia.com/CorporateBond/SearchISIN.aspx)
        - [NSE India](https://www.nseindia.com)
        - [BSE India](https://www.bseindia.com)
        """,
    )

# ======================== Fund Raising UI (Simplified Charts) ========================
def render_fund_raising_panel(default_url: str, segment_label: str):
    st.subheader("Data Source")
    data_url = st.text_input("Paste a public URL (Google Sheet / CSV / XLSX / JSON / HTML table):", value=default_url, key=f"fund_url_{segment_label}")

    if not data_url.strip():
        st.warning("Please provide a data URL.")
        st.stop()

    try:
        df = load_fundraising_url(data_url.strip(), segment=segment_label)
    except Exception as e:
        st.error(f"Could not read the URL. Make sure it’s publicly accessible.\n\nDetails: {e}")
        st.stop()

    # Build filters (Entity, FY, Type, Category)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            entities = sorted(df[ENT_COL].dropna().astype(str).unique())
            ent_sel = st.multiselect("Entity", entities, default=entities)
        with c2:
            fy_vals = sorted(df[FY_COL].dropna().astype(str).unique())
            fy_sel = st.multiselect("Financial Year", fy_vals, default=fy_vals)
        with c3:
            type_col = df.attrs["__fund_cols__"]["type"]
            types = sorted(df[type_col].dropna().astype(str).unique()) if type_col else []
            type_sel = st.multiselect("Type of Issue", types, default=types)
        with c4:
            cat_col = df.attrs["__fund_cols__"]["category"]
            cats = sorted(df[cat_col].dropna().astype(str).unique()) if cat_col else []
            cat_sel = st.multiselect("Category", cats, default=cats)

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if ent_sel:
        mask &= df[ENT_COL].astype(str).isin(ent_sel)
    if fy_sel:
        mask &= df[FY_COL].astype(str).isin(fy_sel)
    if type_sel and df.attrs["__fund_cols__"]["type"]:
        mask &= df[df.attrs["__fund_cols__"]["type"]].astype(str).isin(type_sel)
    if cat_sel and df.attrs["__fund_cols__"]["category"]:
        mask &= df[df.attrs["__fund_cols__"]["category"]].astype(str).isin(cat_sel)

    fdf = df[mask].copy()
    if fdf.empty:
        st.warning("No rows match your filters.")
        st.stop()

    # ---------- KPIs ----------
    total_amount = fdf["Amount of Fund Raised (num)"].sum(skipna=True)
    num_raises   = int(fdf.shape[0])
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f'<div class="kpi">💰 <b>Total Amount Raised</b><br><span style="font-size:26px;font-weight:700;">{total_amount:,.2f}</span><br><span class="muted">(units as in data)</span></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi">📈 <b>Number of Times Fund Raised</b><br><span style="font-size:26px;font-weight:700;">{num_raises}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------- Charts (only Entity, Amount, FY) ----------
    # 1) Amount by Financial Year
    fy_chart_df = fdf.groupby(FY_COL, as_index=False)["Amount of Fund Raised (num)"].sum()
    if not fy_chart_df.empty:
        c_fy = (
            alt.Chart(fy_chart_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{FY_COL}:N", sort=None, title="Financial Year"),
                y=alt.Y("Amount of Fund Raised (num):Q", title="Amount (as in data)"),
                tooltip=[alt.Tooltip(FY_COL, title="FY"), alt.Tooltip("Amount of Fund Raised (num)", title="Amount", format=",")]
            )
            .properties(height=280)
        )
        st.altair_chart(c_fy, use_container_width=True)

    # 2) Amount by Entity (Top 10)
    ent_chart_df = (
        fdf.groupby(ENT_COL, as_index=False)["Amount of Fund Raised (num)"]
        .sum()
        .sort_values("Amount of Fund Raised (num)", ascending=False)
        .head(10)
    )
    if not ent_chart_df.empty:
        c_ent = (
            alt.Chart(ent_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Amount of Fund Raised (num):Q", title="Amount (as in data)"),
                y=alt.Y(f"{ENT_COL}:N", sort='-x', title="Entity"),
                tooltip=[alt.Tooltip(ENT_COL, title="Entity"), alt.Tooltip("Amount of Fund Raised (num)", title="Amount", format=",")]
            )
            .properties(height=320)
        )
        st.altair_chart(c_ent, use_container_width=True)

    st.markdown("---")

    # ---------- Table (with filters applied) ----------
    cols = df.columns
    listed_on = df.attrs["__fund_cols__"]["listed_on"]
    pp_listed = df.attrs["__fund_cols__"]["public_private"]
    date_fmt  = "Date of Fund raising (fmt)"
    type_col  = df.attrs["__fund_cols__"]["type"]
    cat_col   = df.attrs["__fund_cols__"]["category"]

    show_cols = [ENT_COL, FY_COL, date_fmt]
    for optional in [listed_on, pp_listed, type_col, cat_col]:
        if optional and optional not in show_cols:
            show_cols.append(optional)
    show_cols += ["Amount of Fund Raised (num)", "No. of Units Issued (num)", "Unit Capital at End (num)"]

    table_df = fdf[show_cols].rename(columns={
        ENT_COL: "Entity",
        FY_COL: "Financial Year",
        "Amount of Fund Raised (num)": "Amount of Fund Raised",
        "No. of Units Issued (num)": "No. of Units Issued",
        "Unit Capital at End (num)": "Unit Capital at End"
    })

    # Format numeric columns for display
    for colname in ["Amount of Fund Raised", "No. of Units Issued", "Unit Capital at End"]:
        if colname in table_df.columns:
            table_df[colname] = table_df[colname].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

    st.markdown("### Records")
    st.dataframe(table_df, use_container_width=True)

    # Download filtered data as CSV (unformatted numerics for analysis)
    export_df = fdf.rename(columns={ENT_COL: "Entity", FY_COL: "Financial Year"})
    st.download_button(
        "Download filtered data (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{segment_label.lower()}_fund_raising_filtered.csv",
        mime="text/csv"
    )

# ============================== UI ==============================
st.markdown(
    """
    <div class="app-hero">
      <div class="big-title">REIT / InvIT Dashboard</div>
      <div class="subtle">Monitor Borrowings, Fund Raising, and NDCF with clean, formal visuals.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_fund, tab_borrow, tab_ndcf = st.tabs(["Fund Raising", "Borrowings", "NDCF"])

# ---------------- Fund Raising ----------------
with tab_fund:
    st.header("Fund Raising")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_fund")
    if segment == "InvIT":
        render_fund_raising_panel(DEFAULT_INVIT_FUND_URL, "InvIT")
    else:
        render_fund_raising_panel(DEFAULT_REIT_FUND_URL, "REIT")

# ---------------- Borrowings ----------------
def render_borrowings_entry():
    st.header("Borrowings")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_borrow")
    if segment == "InvIT":
        render_borrowings_panel(DEFAULT_INVIT_BORR_URL, "InvIT", ruleset="invit")
    else:
        render_borrowings_panel(DEFAULT_REIT_BORR_URL, "REIT", ruleset="reit")

with tab_borrow:
    render_borrowings_entry()

# ---------------- NDCF ----------------
with tab_ndcf:
    st.header("NDCF")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_ndcf")
    st.info(f"{segment} NDCF dashboard will appear here once data is available.")
