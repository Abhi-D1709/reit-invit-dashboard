# tabs/sponsor_holding.py
import math
import pandas as pd
import streamlit as st
from datetime import date

from utils.common import (
    ENT_COL, FY_COL, EPS,
    DEFAULT_REIT_SPON_URL, DEFAULT_INVIT_SPON_URL,
    _find_col, _num_series, _standardize_selector_columns, _to_date,
    load_table_url,
)

# ---------- helpers ----------

def _fy_end_date(fy: str) -> date:
    """'2019-20' -> 2020-03-31, '2024-25' -> 2025-03-31, '2020' -> 2020-03-31 (best-effort)."""
    if not isinstance(fy, str):
        fy = str(fy or "")
    fy = fy.strip()
    try:
        if "-" in fy:
            a, b = fy.split("-", 1)
            end_year = int("20" + b[-2:]) if len(b) == 2 else int(b)
        else:
            end_year = int(fy)
        return date(end_year, 3, 31)
    except Exception:
        return date.today()

def _years_between(d1: date, d2: date) -> float:
    return abs((d2 - d1).days) / 365.25

@st.cache_data(show_spinner=False, ttl=300)
def _load_sponsor_df(url: str) -> pd.DataFrame:
    """
    Load the Sponsor Holding sheet and compute Sponsor+Group % and Public %.
    """
    raw = load_table_url(url)
    df = _standardize_selector_columns(raw)

    cols = df.columns
    list_col  = _find_col(cols, aliases=["Date of Listing", "Date of Listing of REIT", "Listing Date"])
    spon_col  = _find_col(cols, must_tokens=["sponsor", "no", "unit"]) or \
                _find_col(cols, aliases=["No. of Units Held by the Sponsor (At the end of FY)"])
    group_col = _find_col(cols, must_tokens=["sponsor", "group", "unit"]) or \
                _find_col(cols, aliases=["No. of Units held by the Sponsor Group (At the end of FY)"])
    total_col = _find_col(cols, must_tokens=["total", "outstanding", "unit"]) or \
                _find_col(cols, aliases=[
                    "Total Outstanding Units of the REIT at the end of FY",
                    "Total Outstanding Units of the Trust at the end of FY",
                ])

    df["__sponsor_units__"]       = _num_series(df, spon_col, 0.0)
    df["__sponsor_group_units__"] = _num_series(df, group_col, 0.0)
    df["__total_units__"]         = _num_series(df, total_col, math.nan)

    if list_col:
        df["Listing Date (display)"] = df[list_col].apply(_to_date)
        df["__listing_dt__"] = pd.to_datetime(df[list_col], errors="coerce", dayfirst=True).dt.date
    else:
        df["Listing Date (display)"] = "-"
        df["__listing_dt__"] = pd.NaT

    tot  = df["__total_units__"].replace({0.0: math.nan})
    hold = df["__sponsor_units__"].add(df["__sponsor_group_units__"], fill_value=0.0)

    df["Sponsor+Group %"] = hold.divide(tot).astype(float)
    df["Public %"]        = (1.0 - df["Sponsor+Group %"]).astype(float)

    df.attrs["__matched_cols__"] = {
        "listing": list_col,
        "sponsor": spon_col,
        "sponsor_group": group_col,
        "total": total_col,
    }
    return df

def _sort_fy(values):
    return sorted(values, key=lambda fy: _fy_end_date(str(fy)))

def _stacked_meter_html(s_pct: float, p_pct: float) -> str:
    """
    Single stacked bar:
    left = Sponsor+Group (blue), right = Public (green).
    Legend shows swatches and icons with percentages.
    """
    s = 0.0 if math.isnan(s_pct) else max(0.0, min(1.0, s_pct))
    # If public pct is NaN, infer from sponsor so bar always sums ~100%
    p = (1.0 - s) if math.isnan(p_pct) else max(0.0, min(1.0, p_pct))

    return f"""
<style>
.sp-meter .legend {{
  display:flex; gap:28px; align-items:center; margin:0 0 10px 0; color:#1f2937;
  font-size:1.02rem; font-weight:700;
}}
.sp-meter .swatch {{
  width:12px; height:12px; display:inline-block; border-radius:3px; margin:0 8px;
  vertical-align:middle;
}}
.sp-meter .sponsor-swatch {{ background:#2F80ED; }}
.sp-meter .public-swatch  {{ background:#27AE60; }}
.sp-meter .pct {{ font-weight:800; margin-left:8px; color:#111827; }}
.sp-meter .track {{
  position: relative; height:16px; background:#e9eef5; border-radius:10px; overflow:hidden;
}}
.sp-meter .seg {{ position:absolute; top:0; height:100%; }}
.sp-meter .sponsor {{ left:0; background:#2F80ED; }}
.sp-meter .public  {{ left:{s*100:.2f}%; background:#27AE60; }}
</style>
<div class="sp-meter">
  <div class="legend">
    <div>🏛️ <span class="swatch sponsor-swatch"></span> Sponsor + Group
      <span class="pct">{s*100:.2f}%</span>
    </div>
    <div>👥 <span class="swatch public-swatch"></span> Public
      <span class="pct">{p*100:.2f}%</span>
    </div>
  </div>
  <div class="track">
    <div class="seg sponsor" style="width:{s*100:.2f}%"></div>
    <div class="seg public"  style="width:{p*100:.2f}%"></div>
  </div>
</div>
    """

# ---------- UI render ----------

def render():
    st.header("Sponsor Holding")

    seg = st.selectbox("Select Segment", ["REIT", "InvIT"], key="sp_seg")
    default_url = DEFAULT_INVIT_SPON_URL if seg == "InvIT" else DEFAULT_REIT_SPON_URL

    st.subheader("Data Source")
    st.caption("Paste a public URL (Google Sheet / CSV / XLSX / JSON / HTML table).")
    url = st.text_input("Data URL", value=default_url, key=f"sp_url_{seg}")

    if not url.strip():
        st.warning("Please provide a data URL.")
        st.stop()

    try:
        df = _load_sponsor_df(url.strip())
    except Exception as e:
        st.error(f"Could not read the URL. Make sure it’s publicly accessible.\n\nDetails: {e}")
        st.stop()

    if df.empty or df[ENT_COL].isna().all():
        st.warning("Couldn’t find entity names in the sheet. Ensure a 'Name of REIT/InvIT' column is present.")
        st.stop()

    ent = st.selectbox("Entity", sorted(df[ENT_COL].dropna().astype(str).unique()), key="sp_ent")

    fy_opts = df.loc[df[ENT_COL] == ent, FY_COL].dropna().astype(str).unique().tolist()
    if not fy_opts:
        st.warning("No financial years found for the selected entity.")
        st.stop()

    fy = st.selectbox("Financial Year", _sort_fy(fy_opts), key="sp_fy")

    row_df = df[(df[ENT_COL] == ent) & (df[FY_COL] == fy)]
    if row_df.empty:
        st.warning("No data for the selected filters."); st.stop()
    row = row_df.iloc[0]

    s_pct = float(row.get("Sponsor+Group %", float("nan")))
    p_pct = float(row.get("Public %", float("nan")))

    # Title & single stacked bar with color legend
    st.markdown("### Unitholding")
    st.markdown(_stacked_meter_html(s_pct, p_pct), unsafe_allow_html=True)

    # ---------- Breakup ----------
    m = df.attrs.get("__matched_cols__", {})
    st.markdown("### Breakup")
    def _fmt(x):
        try:
            return f"{float(str(x).replace(',','')):,.2f}"
        except Exception:
            return str(x)

    st.write(
        f"- **Sponsor Units:** {_fmt(row.get(m.get('sponsor') or 'Sponsor Units', '-'))}\n"
        f"- **Sponsor Group Units:** {_fmt(row.get(m.get('sponsor_group') or 'Sponsor Group Units', '-'))}\n"
        f"- **Total Outstanding Units:** {_fmt(row.get(m.get('total') or 'Total Outstanding Units', '-'))}\n"
    )
    st.markdown("---")

    # ---------- Compliance checks ----------
    list_dt = row.get("__listing_dt__", None)
    list_dt_display = row.get("Listing Date (display)", "-")
    fy_end = _fy_end_date(fy)

    within_3yrs = False
    if isinstance(list_dt, date):
        within_3yrs = (_years_between(list_dt, fy_end) < 3.0)

    # Rule 1: first 3 years — Sponsor+Group ≥ 15%
    if within_3yrs:
        if not math.isnan(s_pct) and (s_pct + EPS) < 0.15:
            st.error(f"ALERT: Within first 3 years of listing — Sponsor+Group holding is {s_pct*100:.2f}% (< 15%).")
        else:
            st.success("Within first 3 years of listing — Sponsor+Group requirement (≥ 15%) satisfied.")
    # Rule 2: after 3 years — Public ≥ 25%
    else:
        if not math.isnan(p_pct) and (p_pct + EPS) < 0.25:
            st.error(f"ALERT: After 3 years — Public holding is {p_pct*100:.2f}% (< 25%).")
        else:
            st.success("After 3 years — minimum public unitholding (≥ 25%) satisfied.")

    st.caption(
        f"Listing date: **{list_dt_display}**  •  FY end considered: **{fy_end.isoformat()}**  •  "
        f"{'Within' if within_3yrs else 'Beyond'} first 3 years window."
    )
