# app.py
import io
import re
import math
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

# --------------------------- Page Config & Light Styling ---------------------------
st.set_page_config(page_title="REIT / InvIT Dashboard", page_icon="📊", layout="wide")

# Subtle, formal UI styling (works with Streamlit light & dark themes)
st.markdown(
    """
    <style>
      .app-hero {
        padding: 14px 18px;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.06);
        background: linear-gradient(180deg, rgba(25,118,210,0.08) 0%, rgba(25,118,210,0.03) 100%);
        margin-bottom: 14px;
      }
      .big-title { font-size: 1.9rem; font-weight: 700; margin: 0; line-height: 1.2; }
      .subtle { color: var(--text-color-secondary, #6b7280); margin-top: 6px; }
      .card {
        padding: 14px 16px; border-radius: 12px; background: rgba(255,255,255,0.7);
        border: 1px solid rgba(0,0,0,0.06);
      }
      .kpi {
        padding: 12px 14px; border-radius: 12px; color: #fff;
        background: linear-gradient(135deg, #1976D2, #115293);
      }
      .chip {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        font-size: 12px; border: 1px solid rgba(0,0,0,0.12);
        background: rgba(0,0,0,0.03);
        margin-left: 8px;
      }
      .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
      .muted { color: #6b7280; }
      .link-list a { text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Defaults ---------------------------
# Your Google Sheet (view link is fine). You can paste any other URL in the UI.
DEFAULT_DATA_URL = "https://docs.google.com/spreadsheets/d/1OugwmVbR2BXjWcRGOlLhqrg3APVv9R17LYpZPDeFDkw/edit?usp=sharing"

# Canonical internal names (we map your sheet headers to these)
ENT_COL = "__Entity__"
FY_COL  = "__FinancialYear__"
QTR_COL = "__QuarterEnded__"

# --------------------------- Helpers ---------------------------
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

YES_PAT = re.compile(r"\b(yes|y|true|approved|taken)\b|^1$|✓", re.I)
def _is_yes(value):
    if value is None or pd.isna(value):
        return False
    return YES_PAT.search(str(value).strip()) is not None

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

def _num_series(df: pd.DataFrame, colname: str, fill=0.0) -> pd.Series:
    """Return numeric Series; if col missing, create Series with `fill`."""
    if colname and colname in df.columns:
        return pd.to_numeric(df[colname], errors="coerce")
    fv = np.nan if (fill is pd.NA or fill is None) else fill
    return pd.Series([fv] * len(df), index=df.index, dtype="float64")

def _standardize_selector_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map your sheet’s headers to canonical Entity / Financial Year / Quarter Ended."""
    cols = df.columns
    entity_col = _find_col(
        cols,
        aliases=["Entity", "Entity Name", "REIT", "REIT Name", "Name of REIT",
                 "Trust", "Trust Name", "Issuer", "Issuer Name", "InvIT / REIT Name"]
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

    missing = []
    if entity_col is None: missing.append("Entity (e.g., 'Entity' / 'REIT Name')")
    if fy_col is None:     missing.append("Financial Year (e.g., 'Financial Year' / 'FY')")
    if qtr_col is None:    missing.append("Quarter Ended (e.g., 'Quarter Ended' / 'Quarter')")

    if missing:
        df.attrs["__fatal__"] = (
            "Missing required selector columns:\n- " + "\n- ".join(missing) +
            "\n\nAvailable columns:\n- " + "\n- ".join(map(str, cols))
        )
        # prevent downstream KeyError before we stop:
        for cname in (ENT_COL, FY_COL, QTR_COL):
            if cname not in df.columns:
                df[cname] = np.nan
        return df

    df = df.rename(columns={entity_col: ENT_COL, fy_col: FY_COL, qtr_col: QTR_COL})
    df.attrs["__selector_map__"] = {"Entity": entity_col, "Financial Year": fy_col, "Quarter Ended": qtr_col}
    return df

def _quarter_sort(values):
    order = {"June": 0, "Sept": 1, "Sep": 1, "December": 2, "Dec": 2, "Mar": 3, "March": 3}
    return sorted(values, key=lambda v: order.get(str(v), 99))

def _share_to_csv_url(url: str) -> str:
    """
    Accepts:
      - a Google Sheets view URL like .../edit?usp=sharing or .../edit#gid=123
      - already /export?format=csv&gid=...
      - published .../pub?output=csv
    Returns a /export?format=csv&gid=... URL.
    """
    if not url:
        return ""
    if "output=csv" in url or "/export" in url:
        return url  # already CSV export
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        return url  # not a sheet; caller will handle other formats
    sheet_id = m.group(1)
    gid = "0"
    if "#gid=" in url:
        gid = url.split("#gid=")[-1].split("&")[0]
    else:
        qs_gid = parse_qs(urlparse(url).query).get("gid", [None])[0]
        if qs_gid:
            gid = str(qs_gid)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# --------------------------- Processing logic (shared) ---------------------------
def _process_borrowings_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = _standardize_selector_columns(df)

    cols = df.columns
    borrow_col = _find_col(cols, aliases=["Borrowings","A. Borrowings","A - Borrowings","A Borrowings"], must_tokens=["borrow"])
    defer_col  = _find_col(cols, aliases=["Deferred Payments","B. Deferred Payments","Deferred Payment"], must_tokens=["defer","payment"])
    cash_col   = _find_col(cols, aliases=["Cash and Cash Equivalents","C. Cash and Cash Equivalents","Cash & Cash Equivalents","Cash and cash equivalents"], must_tokens=["cash","equivalent"])
    assets_col = _find_col(cols, aliases=["Value of REIT Assets","D. Value of REIT Assets","Value of Assets"], must_tokens=["value","asset","reit"])

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
        "Cash and Cash Equivalents": cash_col, "Value of REIT Assets": assets_col,
        "NBR source": nbr_col or "computed",
    }
    return df

# --------------------------- Universal URL Loader ---------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_borrowings_url(url: str) -> pd.DataFrame:
    """
    Accepts public URLs for:
      - Google Sheets (view link auto-converted to CSV export)
      - CSV
      - XLSX
      - JSON
      - HTML pages containing <table>
    Returns a cleaned dataframe ready for the dashboard.
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
            return _process_borrowings_df(df)
    except Exception:
        pass

    # 2) Try Excel via bytes
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=25)
        resp.raise_for_status()
        # Try as excel
        try:
            df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _process_borrowings_df(df)
        except Exception:
            pass
        # 3) Try JSON from same response
        try:
            data = resp.json()
            # Normalize nested JSON into a DataFrame (best-effort)
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                # choose a likely records key or flatten dict
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
                return _process_borrowings_df(df)
        except Exception:
            pass
    except Exception:
        pass

    # 4) Try JSON via pandas (URL)
    try:
        df = pd.read_json(url)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return _process_borrowings_df(df)
    except Exception:
        pass

    # 5) Try HTML tables
    try:
        tables = pd.read_html(url)
        if tables:
            df = max(tables, key=lambda t: (t.shape[0] * t.shape[1]))
            return _process_borrowings_df(df)
    except Exception:
        pass

    raise ValueError("Couldn't parse the URL as CSV, Excel, JSON, or an HTML table. "
                     "Ensure it is publicly accessible without login.")

# ============================== UI ==============================
st.markdown(
    """
    <style>
      /* add/override after your existing CSS */
      .app-hero {
        text-align: center;
        max-width: 900px;          /* tweak as you like */
        margin: 0 auto 14px;       /* centers the block */
      }
      .big-title, .subtle { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    st.info(f"{segment} Fund Raising dashboard will appear here once data is available.")

# ---------------- Borrowings ----------------
with tab_borrow:
    st.header("Borrowings")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_borrow")

    if segment == "InvIT":
        st.info("InvIT Borrowings dashboard will appear here once data is available.")
    else:
        st.subheader("Data Source")
        data_url = st.text_input("Paste a public URL (Google Sheet / CSV / XLSX / JSON / HTML table):", value=DEFAULT_DATA_URL, placeholder="https://...")

        if not data_url.strip():
            st.warning("Please provide a data URL.")
            st.stop()

        try:
            df = load_borrowings_url(data_url.strip())
        except Exception as e:
            st.error(f"Could not read the URL. Make sure it’s publicly accessible.\n\nDetails: {e}")
            st.stop()

        fatal = df.attrs.get("__fatal__")
        if fatal:
            st.error(fatal)
            st.stop()

        # Filters (inside a card)
        with st.container():
            filt_cols = st.columns(3)
            with filt_cols[0]:
                entities = sorted(df[ENT_COL].dropna().astype(str).unique())
                entity = st.selectbox("Entity", entities)
            with filt_cols[1]:
                fy_options = sorted(df.loc[df[ENT_COL] == entity, FY_COL].dropna().astype(str).unique())
                fy = st.selectbox("Financial Year", fy_options)
            with filt_cols[2]:
                qtr_present = df.loc[(df[ENT_COL] == entity) & (df[FY_COL] == fy), QTR_COL].dropna().astype(str).unique().tolist()
                qtr = st.selectbox("Quarter", _quarter_sort(qtr_present))

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
            st.markdown('<div class="kpi">📊 <b>Net Borrowings Ratio</b><br>'
                        f'<span style="font-size:28px;font-weight:700;">{nbr_display}</span></div>', unsafe_allow_html=True)
            if isinstance(nbr, (int, float)) and not pd.isna(nbr):
                st.progress(min(max(float(nbr), 0.0), 1.0))

        with colB:
            m = df.attrs.get("__matched_cols__", {})
            a_label = m.get("Borrowings") or "Borrowings"
            b_label = m.get("Deferred Payments") or "Deferred Payments"
            c_label = m.get("Cash and Cash Equivalents") or "Cash and Cash Equivalents"
            d_label = m.get("Value of REIT Assets") or "Value of REIT Assets"
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

        # Conditional sections
        SHOW_THRESHOLD = 0.25
        show_sections = isinstance(nbr, (int, float)) and not pd.isna(nbr) and nbr >= SHOW_THRESHOLD

        if not show_sections:
            st.info("NBR is below 25%. Credit Rating, Unitholder Approval, and Additional Compliances are not required to be displayed.")
        else:
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

            cra1_rating = row.get(cra1_rating_col)
            cra2_rating = row.get(cra2_rating_col)
            cra1_name   = row.get(cra1_name_col)
            cra2_name   = row.get(cra2_name_col)
            cra1_date   = row.get(cra1_date_col)
            cra2_date   = row.get(cra2_date_col)
            cra1_link   = _url(row.get(cra1_link_col))
            cra2_link   = _url(row.get(cra2_link_col))

            credit_taken = _is_taken(cra1_rating) or _is_taken(cra2_rating)

            # Unitholder Approval
            ua_col = _find_col(
                cols,
                aliases=["Unitholder Approval", "Unitholder approval"],
                must_tokens=["unitholder", "approval"],
                exclude_tokens=["date", "meeting", "weblink", "notice", "votes", "record", "favour", "against", "total"]
            )
            unitholder_approval_val = row.get(ua_col)
            unit_taken = _is_yes(unitholder_approval_val)

            # Alerts
            missing = []
            if not credit_taken:
                missing.append("Credit Rating")
            if not unit_taken:
                missing.append("Unitholder Approval")
            if missing:
                msg = ("Both Credit Rating and Unitholder Approval are not taken / not available for this period, even though NBR ≥ 25%."
                       if len(missing) == 2 else
                       f"{missing[0]} is not taken / not available for this period, even though NBR ≥ 25%."
                )
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
            approval_display = "Yes" if unit_taken else ("No" if _is_taken(unitholder_approval_val) else "-")
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

# ---------------- NDCF ----------------
with tab_ndcf:
    st.header("NDCF")
    segment = st.selectbox("Select Segment", ["REIT", "InvIT"], key="seg_ndcf")
    st.info(f"{segment} NDCF dashboard will appear here once data is available.")
