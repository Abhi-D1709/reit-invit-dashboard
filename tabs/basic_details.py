# tabs/directory.py
from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from utils.common import (
    load_table_url,                # uses your existing helper
    DEFAULT_REIT_DIR_URL,
    DEFAULT_INVIT_DIR_URL,
)

# =========================== small utilities ===========================

def _clean(s) -> str:
    if s is None:
        return ""
    s = str(s).replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", s)

def _split_list(val) -> List[str]:
    """Split multi-line / comma / semicolon sponsor lists; strip bullets like '1. '."""
    if val is None or str(val).strip() == "":
        return []
    s = str(val).replace("\r", "\n")
    s = re.sub(r"\n?\s*\d+\.\s*", "\n", s)          # remove "1. " bullets
    parts = re.split(r"[\n;|,]+", s)
    return [p.strip() for p in parts if p and p.strip() != "-"]

def _popover_or_expander(label: str):
    """Use popover if available (Streamlit ≥ 1.31), else expander."""
    return st.popover(label, use_container_width=True) if hasattr(st, "popover") else st.expander(label, expanded=False)

def _details_md(items: Dict[str, str]) -> str:
    lines = []
    for k, v in items.items():
        v = _clean(v) if isinstance(v, str) else v
        if v and str(v).strip() != "-":
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) if lines else "- No additional details available."

def _first_nonempty(df: pd.DataFrame, row: pd.Series, candidates: List[str]) -> Optional[str]:
    """Return the first column name in candidates that has a non-empty value in this row."""
    for c in candidates:
        if c in df.columns:
            v = row.get(c)
            if v is not None and str(v).strip() not in ("", "-"):
                return c
    return None

def _sheet_id_from_url(url: str) -> Optional[str]:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None

def _gid_from_url(url: str) -> Optional[str]:
    m = re.search(r"[?&]gid=(\d+)", url)
    return m.group(1) if m else None


# ============================== loaders ===============================

@st.cache_data(ttl=300, show_spinner=False)
def _load_sheet1(url: str) -> pd.DataFrame:
    df = load_table_url(url)
    df.columns = [_clean(c) for c in df.columns]
    return df

def _looks_like_sheet2(df: pd.DataFrame) -> bool:
    """Return True if df appears to be the FY-wise SPV/Holdco table."""
    if df is None or df.empty:
        return False
    cols = {re.sub(r"[^a-z0-9]+", "", c.lower()) for c in df.columns}
    hints = [
        "financialyear", "fy",
        "nameofspv", "spv",
        "nameofholdco", "holdco",
    ]
    return any(h in cols for h in hints)

@st.cache_data(ttl=300, show_spinner=False)
def _load_sheet2(url: str) -> pd.DataFrame:
    """
    Load the FY-wise tab (Sheet2) robustly.

    Strategy:
      - Try explicit gid from the URL via CSV endpoints.
      - If what we get *doesn't* look like the FY table, keep trying by sheet name:
        ("Sheet2", "Sheet 2", "FY", "SPVs", "SPV Map", "FY Map")
      - Only return a DataFrame once it looks like the FY table.
      - Else return an empty DF with expected columns.
    """
    expected_cols = ["Financial Year", "Name of REIT", "Name of SPV", "Name of Holdco"]
    sid = _sheet_id_from_url(url)
    if not sid:
        return pd.DataFrame(columns=expected_cols)

    candidates: List[str] = []

    # From gid in URL
    gid = _gid_from_url(url)
    if gid:
        candidates += [
            f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}",
            f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&gid={gid}",
        ]

    # From common sheet names
    for nm in ["Sheet2", "Sheet 2", "FY", "SPVs", "SPV Map", "FY Map"]:
        candidates += [
            f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&sheet={nm}",
            f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={nm}",
        ]

    for u in candidates:
        try:
            df = pd.read_csv(u)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.columns = [_clean(c) for c in df.columns]
                if _looks_like_sheet2(df):
                    return df
        except Exception:
            pass

    # Nothing matched—return empty with expected headers
    return pd.DataFrame(columns=expected_cols)


# =========================== column maps ==============================

def _map_sheet1(df: pd.DataFrame) -> Dict[str, List[str]]:
    def cands(*names) -> List[str]:
        out = []
        cols = df.columns
        norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in cols}
        for n in names:
            if n in cols:
                out.append(n)
            nkey = re.sub(r"[^a-z0-9]+", "", n.lower())
            if nkey in norm and norm[nkey] not in out:
                out.append(norm[nkey])
        return out

    return {
        "entity":   cands("Name of REIT", "Name of InvIT", "Trust", "Name", "Issuer", "Entity"),
        "reit_addr":cands("Address of Principal Place of Business of REIT",
                          "Address of Principal Place of Business of InvIT",
                          "Registered Address", "Address"),
        "reg_date": cands("Date of SEBI registration", "SEBI Registration Date"),
        "reg_no":   cands("Registration No.", "SEBI Registration No"),
        "co_name":  cands("Name of the Compliance Officer", "Compliance Officer"),
        "co_email": cands("Email ID of the Compliance Officer", "Compliance Officer Email"),
        "co_mobile":cands("Mobile No of the Compliance Officer", "Compliance Officer Mobile"),
        "sponsors": cands("Name of current Sponsor(s)", "Sponsors"),
        "manager":  cands("Name of current Manager", "Manager"),
        "mgr_addr": cands("Address of the Manager", "Manager Address"),
        "trustee":  cands("Name of current Trustee", "Trustee"),
        "tr_addr":  cands("Address of the Trustee", "Trustee Address"),
        "tr_regno": cands("SEBI registration No of the Trustee", "Trustee SEBI Registration No"),
    }

def _map_sheet2(df: pd.DataFrame) -> Dict[str, List[str]]:
    def cands(*names) -> List[str]:
        out = []
        cols = df.columns
        norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in cols}
        for n in names:
            if n in cols:
                out.append(n)
            nkey = re.sub(r"[^a-z0-9]+", "", n.lower())
            if nkey in norm and norm[nkey] not in out:
                out.append(norm[nkey])
        return out

    return {
        "fy":       cands("Financial Year", "FY", "Financial Yr"),
        "entity":   cands("Name of REIT", "Name of InvIT", "Entity", "Trust"),
        "spv":      cands("Name of SPV", "SPV", "Subsidiary"),
        "holdco":   cands("Name of Holdco", "Holdco", "Holding Company", "Holding Co"),
    }


# ============================== render ================================

def render():
    st.header("Basic Details")

    with st.sidebar:
        seg = st.selectbox(
            "Select Segment",
            ["REIT", "InvIT"],
            index=0,
            key="bd_segment"
        )
        default_url = DEFAULT_REIT_DIR_URL if seg == "REIT" else DEFAULT_INVIT_DIR_URL
        url = st.text_input(
            "Data URL (Google Sheet - public view)",
            value=default_url or "",
            placeholder="Paste a public Google Sheet URL…",
            key=f"bd_url_{seg}"
        )

    # ---- Sheet1: core directory ----
    try:
        df1 = _load_sheet1(url)
    except Exception as e:
        st.error(f"Could not load Sheet1. {type(e).__name__}: {e}")
        return

    # ---- Sheet2: FY-wise SPV/HoldCo (auto-detected) ----
    df2 = _load_sheet2(url)

    if df1.empty:
        st.warning("Sheet1 appears empty.")
        return

    # Map directory columns & entity list
    m1 = _map_sheet1(df1)
    ent_col = _first_nonempty(df1, df1.iloc[0], m1["entity"]) or (m1["entity"][0] if m1["entity"] else None)
    if not ent_col or ent_col not in df1.columns:
        st.error("Could not find the entity name column in Sheet1 (e.g., 'Name of REIT').")
        return

    entities = [e for e in df1[ent_col].astype(str).fillna("").tolist() if e and e.strip() and e.strip() != "-"]
    ent = st.selectbox("Choose entity", sorted(entities))

    # Extract main (static) details from Sheet1
    row1 = df1.loc[df1[ent_col].astype(str) == ent]
    if row1.empty:
        st.warning("Selected entity not found in Sheet1.")
        return
    row1 = row1.iloc[0]

    sponsors = _split_list(row1.get(_first_nonempty(df1, row1, m1["sponsors"]) or ""))
    manager  = _clean(row1.get(_first_nonempty(df1, row1, m1["manager"]) or ""))
    trustee  = _clean(row1.get(_first_nonempty(df1, row1, m1["trustee"]) or ""))

    mgr_addr = _clean(row1.get(_first_nonempty(df1, row1, m1["mgr_addr"]) or ""))
    tr_addr  = _clean(row1.get(_first_nonempty(df1, row1, m1["tr_addr"]) or ""))
    tr_regno = _clean(row1.get(_first_nonempty(df1, row1, m1["tr_regno"]) or ""))

    reit_addr = _clean(row1.get(_first_nonempty(df1, row1, m1["reit_addr"]) or ""))
    reg_date  = _clean(row1.get(_first_nonempty(df1, row1, m1["reg_date"]) or ""))
    reg_no    = _clean(row1.get(_first_nonempty(df1, row1, m1["reg_no"]) or ""))

    co_name   = _clean(row1.get(_first_nonempty(df1, row1, m1["co_name"]) or ""))
    co_email  = _clean(row1.get(_first_nonempty(df1, row1, m1["co_email"]) or ""))
    co_mobile = _clean(row1.get(_first_nonempty(df1, row1, m1["co_mobile"]) or ""))

    # ---------- Prepare FY options for chosen entity (from Sheet2) ----------
    fy_options: List[str] = []
    df2_for_ent = pd.DataFrame()
    spv_col = hc_col = None

    if not df2.empty:
        df2.columns = [_clean(c) for c in df2.columns]
        m2 = _map_sheet2(df2)
        fy_col  = next((c for c in m2["fy"] if c in df2.columns), None)
        e2_col  = next((c for c in m2["entity"] if c in df2.columns), None)
        spv_col = next((c for c in m2["spv"] if c in df2.columns), None)
        hc_col  = next((c for c in m2["holdco"] if c in df2.columns), None)

        if fy_col and e2_col:
            df2_for_ent = df2.copy()
            for c in [fy_col, e2_col]:
                df2_for_ent[c] = df2_for_ent[c].astype(str).map(_clean)
            df2_for_ent = df2_for_ent[df2_for_ent[e2_col] == _clean(ent)]
            fy_options = sorted(
                [fy for fy in df2_for_ent[fy_col].dropna().astype(str).map(_clean).unique() if fy],
                key=lambda x: (len(x), x),
            )
        else:
            st.caption("Note: Sheet2 column names were not recognized; FY-wise SPVs/HoldCos not shown.")
            with st.expander("Show Sheet2 columns detected"):
                st.write(list(df2.columns))
    else:
        st.caption("Sheet2 could not be auto-loaded. Open the tab in the browser and copy the link so its gid appears in the URL.")

    # ---------- light card styles ----------
    st.markdown(
        """
        <style>
        .dir-card {
          border: 1px solid rgba(0,0,0,.08);
          background: #0f5e7610;
          border-radius: 12px; padding: 10px 12px; margin-bottom: 8px;
        }
        .dir-title {
          font-size: .92rem; font-weight: 600; color: #0f172a; text-align:center;
          margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- top row: Sponsor | Manager | Trustee ----------
    c1, c2, c3 = st.columns(3, gap="large")

    def _card(title: str, names: List[str], blocks: List[str]):
        st.markdown(f'<div class="dir-card"><div class="dir-title">{title}</div></div>', unsafe_allow_html=True)
        row = st.container()
        cols = row.columns(max(1, min(len(names), 3)))
        for i, name in enumerate(names):
            with cols[i % len(cols)]:
                with _popover_or_expander(name or "—"):
                    st.markdown(blocks[i], unsafe_allow_html=True)

    with c1:
        sponsor_names = sponsors if sponsors else ["—"]
        sponsor_blocks = [_details_md({"Role": "Sponsor", "Entity": s}) for s in sponsor_names]
        _card("Sponsor", sponsor_names, sponsor_blocks)

    with c2:
        mgr_names = [manager] if manager else ["—"]
        mgr_blocks = [_details_md({"Role": "Manager", "Entity": manager or "—", "Address": mgr_addr})]
        _card("Manager", mgr_names, mgr_blocks)

    with c3:
        tr_names = [trustee] if trustee else ["—"]
        tr_blocks = [_details_md({"Role": "Trustee", "Entity": trustee or "—", "Address": tr_addr, "SEBI Registration No": tr_regno})]
        _card("Trustee", tr_names, tr_blocks)

    # ---------- center: Entity card ----------
    with _popover_or_expander(_clean(ent)):
        st.markdown(
            _details_md({
                "Address of Principal Place of Business": reit_addr,
                "SEBI Registration Date": reg_date,
                "SEBI Registration No": reg_no,
                "Compliance Officer": co_name,
                "CO Email": co_email,
                "CO Mobile": co_mobile,
            }),
            unsafe_allow_html=True,
        )

    # ---------- FY selector (just above HoldCos / SPVs) ----------
    spv_list: List[str] = []
    holdco_list: List[str] = []
    holdco_spvs_map: Dict[str, List[str]] = {}

    if fy_options:
        fy = st.selectbox("Financial Year (for SPVs / HoldCos)", fy_options, index=0)

        m2 = _map_sheet2(df2_for_ent if not df2_for_ent.empty else df2)
        fy_col = next((c for c in m2["fy"] if c in (df2_for_ent.columns if not df2_for_ent.empty else df2.columns)), None)

        use_df = df2_for_ent if not df2_for_ent.empty else df2
        if fy_col:
            fy_df = use_df[use_df[fy_col].astype(str).map(_clean) == _clean(fy)]
            if not fy_df.empty:
                if spv_col and spv_col in fy_df.columns:
                    spv_list = [s for s in fy_df[spv_col].fillna("").astype(str).map(_clean) if s and s != "-"]
                if hc_col and hc_col in fy_df.columns:
                    holdco_list = [h for h in fy_df[hc_col].fillna("").astype(str).map(_clean) if h and h != "-"]

                # Map HoldCo -> [SPVs]
                if spv_col and hc_col and spv_col in fy_df.columns and hc_col in fy_df.columns:
                    tmp = fy_df[[hc_col, spv_col]].dropna()
                    for h, group in tmp.groupby(hc_col):
                        h = _clean(h)
                        ss = sorted({_clean(v) for v in group[spv_col].tolist() if _clean(v)})
                        if h:
                            holdco_spvs_map[h] = ss
    else:
        st.info("No FY rows found for this entity in Sheet2 (SPVs/HoldCos will be blank).")

    # ---------- bottom row: HoldCos | SPVs (selected FY) ----------
    c4, c5 = st.columns(2, gap="large")

    with c4:
        if holdco_list:
            h_names = sorted(set(holdco_list))
            blocks = []
            for h in h_names:
                extra = {}
                if h in holdco_spvs_map and holdco_spvs_map[h]:
                    extra["SPVs under this HoldCo"] = ", ".join(holdco_spvs_map[h])
                blocks.append(_details_md({"Role": "HoldCo", "Entity": h, **extra}))
            _card("HoldCos (selected FY)", h_names, blocks)
        else:
            _card("HoldCos (selected FY)", ["—"], [_details_md({})])

    with c5:
        if spv_list:
            # Build reverse map: SPV -> [HoldCo(s)]
            spv_to_holdcos: Dict[str, List[str]] = {}
            for h, s_list in holdco_spvs_map.items():
                for s in s_list:
                    spv_to_holdcos.setdefault(s, []).append(h)

            s_names = sorted(set(spv_list))
            blocks = []
            for s in s_names:
                extra = {}
                if s in spv_to_holdcos and spv_to_holdcos[s]:
                    extra["Part of HoldCo(s)"] = ", ".join(sorted(set(spv_to_holdcos[s])))
                blocks.append(_details_md({"Role": "SPV", "Entity": s, **extra}))
            _card("SPVs (selected FY)", s_names, blocks)
        else:
            _card("SPVs (selected FY)", ["—"], [_details_md({})])
