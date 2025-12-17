# tabs/ndcf.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- Configuration Class --------
@dataclass
class NDCFConfig:
    """Configuration for NDCF column names and constants."""
    
    # Default URLs
    DEFAULT_SHEET_URL_TRUST: str = "https://docs.google.com/spreadsheets/d/18QgoAV_gOQ1ShnVbXzz8bu3V3a1mflevB-foGh27gbA/edit?usp=sharing"
    TRUST_SHEET_NAME: str = "NDCF REITs"
    SPV_SHEET_NAME: str = "NDCF SPV REIT"
    DEFAULT_REIT_DIR_URL: Optional[str] = None
    
    # Trust level columns - matching actual sheet structure
    COMP_COL: str = "Total Amount of NDCF computed as per NDCF Statement"
    DECL_INCL_COL: str = "Total Amount of NDCF declared for the period (incl. Surplus)"
    DECL_EXCL_COL: str = "Total Amount of NDCF declared for the period (excl. Surplus)"
    CFO_COL: str = "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)"
    CFI_COL: str = "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)"
    CFF_COL: str = "Cash Flow From Fincing Activities as per Cash Flow Statements (as per Audited Fincials or Fincials with Limited Review)"
    PAT_COL: str = "Profit after tax as per Statement of Profit and Loss (as per Audited Fincials or Fincials with Limited Review)"
    
    # SPV level columns
    SPV_CFO: str = "SPV Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    SPV_CFI: str = "SPV Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    SPV_CFF: str = "SPV Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    SPV_PAT: str = "SPV Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"
    HCO_CFO: str = "HoldCo Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    HCO_CFI: str = "HoldCo Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    HCO_CFF: str = "Holdco Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
    HCO_PAT: str = "Holdco Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"
    
    # Thresholds
    PAYOUT_THRESHOLD: float = 90.0
    GAP_THRESHOLD: float = 10.0
    RECORD_DATE_DAYS: int = 2
    DISTRIBUTION_DATE_DAYS: int = 5
    
    def __post_init__(self):
        """Override with settings from utils.common if available."""
        try:
            from utils.common import NDCF_REITS_SHEET_URL, DEFAULT_REIT_DIR_URL as _DIR_URL
            if NDCF_REITS_SHEET_URL:
                self.DEFAULT_SHEET_URL_TRUST = NDCF_REITS_SHEET_URL
            if _DIR_URL:
                self.DEFAULT_REIT_DIR_URL = _DIR_URL
        except Exception as e:
            logger.debug(f"Could not load custom config: {e}")
    
    @property
    def trust_numeric_cols(self) -> List[str]:
        """Return list of numeric columns for trust level."""
        return [self.COMP_COL, self.DECL_INCL_COL, self.DECL_EXCL_COL, 
                self.CFO_COL, self.CFI_COL, self.CFF_COL, self.PAT_COL]
    
    @property
    def spv_numeric_cols(self) -> List[str]:
        """Return list of numeric columns for SPV level."""
        return [self.COMP_COL, self.DECL_INCL_COL, self.SPV_CFO, self.SPV_CFI, 
                self.SPV_CFF, self.SPV_PAT, self.HCO_CFO, self.HCO_CFI, 
                self.HCO_CFF, self.HCO_PAT]
    
    @property
    def trust_text_cols(self) -> List[str]:
        """Return list of text columns for trust level."""
        return ["Entity", "Name of REIT", "Financial Year", "Period Ended"]
    
    @property
    def spv_text_cols(self) -> List[str]:
        """Return list of text columns for SPV level."""
        return ["Entity", "Name of REIT", "Financial Year", "Period Ended", 
                "Name of SPV", "Name of Holdco (Leave Blank if N/A)"]

# Initialize config
CONFIG = NDCFConfig()

# -------- Helper Functions --------
def _csv_url_from_gsheet(url: str, sheet: Optional[str] = None, 
                         gid: Optional[str] = None) -> str:
    """Convert Google Sheets URL to CSV export URL."""
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        return url
    sheet_id = m.group(1)
    
    if gid:
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    if sheet:
        from urllib.parse import quote
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet)}"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

def _strip(s) -> str:
    """Strip whitespace from string, handle NaN."""
    if pd.isna(s):
        return ""
    return str(s).strip()

def _to_number(x) -> float:
    """
    Convert value to number, handling various formats.
    
    Handles: commas, parentheses for negatives, empty strings, dashes.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    
    s = str(x).strip()
    if s == "" or s in {"-", "–", "—", "nan", "None"}:
        return np.nan
    
    # Remove commas and spaces
    s = s.replace(",", "").replace(" ", "")
    
    # Handle parentheses as negative
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    
    try:
        return float(s)
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert '{x}' to number: {e}")
        return np.nan

def _to_date(val) -> pd.Timestamp:
    """
    Convert value to date, handling multiple formats.
    
    Handles: Excel serials, DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, MM/DD/YYYY.
    """
    if val is None or val == "":
        return pd.NaT
    
    s = str(val).strip()
    if s == "" or s.lower() in {"none", "null", "na", "-", "nat", "nan"}:
        return pd.NaT
    
    # Try explicit formats first (more reliable)
    date_formats = [
        "%d/%m/%Y",      # DD/MM/YYYY - most common in Indian context
        "%d-%m-%Y",      # DD-MM-YYYY
        "%Y-%m-%d",      # ISO format
        "%d.%m.%Y",      # DD.MM.YYYY
        "%m/%d/%Y",      # MM/DD/YYYY
        "%Y/%m/%d",      # YYYY/MM/DD
        "%d/%m/%y",      # DD/MM/YY
        "%d-%m-%y",      # DD-MM-YY
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(s, fmt)
            return pd.Timestamp(dt.date())
        except ValueError:
            continue
    
    # Handle Excel serial dates
    if re.fullmatch(r"\d{5,6}(\.\d+)?", s):
        try:
            f = float(s)
            if 10000 <= f <= 80000:  # Valid Excel date range
                dt = pd.to_datetime(f, unit="D", origin="1899-12-30", errors="coerce")
                if pd.notna(dt):
                    return pd.Timestamp(dt.date())
        except (ValueError, TypeError):
            pass
    
    # Flexible parse as last resort
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return pd.Timestamp(dt.date())
    except Exception:
        pass
    
    logger.warning(f"Could not parse date '{val}'")
    return pd.NaT

def _status(v: Optional[bool]) -> str:
    """Convert boolean to status emoji."""
    if pd.isna(v) or v is None:
        return "–"
    return "🟢" if bool(v) else "🔴"

def _format_number(val) -> str:
    """Format number for display."""
    if pd.isna(val):
        return "–"
    try:
        num = float(val)
        if abs(num) >= 1000:
            return f"{num:,.2f}"
        elif abs(num) >= 1:
            return f"{num:.2f}"
        else:
            return f"{num:.4f}"
    except:
        return str(val)

# -------- Validation Functions --------
def validate_dataframe(df: pd.DataFrame, required_cols: List[str], 
                      df_type: str = "dataframe") -> Tuple[bool, List[str]]:
    """
    Validate dataframe has required structure.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        df_type: Type description for error messages
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if df.empty:
        errors.append(f"{df_type} is empty")
        return False, errors
    
    # Check for required columns (flexible matching)
    for req_col in required_cols:
        # Try exact match first
        if req_col not in df.columns:
            # Try case-insensitive match
            found = False
            for col in df.columns:
                if col.lower().strip() == req_col.lower().strip():
                    found = True
                    break
            if not found:
                # Only add to errors if it's a critical column
                if any(key in req_col for key in ["REIT", "Financial Year", "NDCF computed"]):
                    errors.append(f"Missing critical column in {df_type}: {req_col}")
    
    # Check for completely empty key columns
    entity_cols = [c for c in df.columns if "reit" in c.lower() or c.lower() == "entity"]
    for col in entity_cols:
        if col in df.columns and df[col].isna().all():
            errors.append(f"Column '{col}' in {df_type} is completely empty")
    
    return len(errors) == 0, errors

# -------- Data Loading Functions --------
@st.cache_data(ttl=3600, show_spinner=False)
def load_reit_ndcf(url: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Load and process REIT NDCF data from Google Sheets.
    
    Args:
        url: Google Sheets URL
        sheet_name: Name of the sheet to load
        
    Returns:
        Processed DataFrame with numeric and date conversions
    """
    if sheet_name is None:
        sheet_name = CONFIG.TRUST_SHEET_NAME
    
    try:
        csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
        df = pd.read_csv(csv_url, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        
        logger.info(f"Loaded columns: {df.columns.tolist()}")
        
        # Rename columns for consistency
        rename_map = {
            "Fincial Year": "Financial Year",
            "Period": "Period Ended",
            "Period ended": "Period Ended",
            "Date of Finalisation/Declaration of NDCF Statement by REIT": "Declaration Date",
            "Date of Filisation/Declaration of NDCF Statement by REIT": "Declaration Date",
            "Date of Finalization/Declaration of NDCF Statement by REIT": "Declaration Date",
            "Record Date": "Record Date",
            "Date of Distribution of NDCF by REIT": "Distribution Date",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Keep "Entity" as primary, create "Name of REIT" as alias if needed
        if "Entity" in df.columns and "Name of REIT" not in df.columns:
            df["Name of REIT"] = df["Entity"]
        elif "Name of REIT" in df.columns and "Entity" not in df.columns:
            df["Entity"] = df["Name of REIT"]
        
        # Keep raw date strings for debugging
        date_related_cols = [c for c in df.columns if "date" in c.lower()]
        for col in date_related_cols:
            if col in df.columns and "__raw" not in col:
                df[f"{col}__raw"] = df[col].copy()
        
        # Convert numeric columns (flexible matching)
        numeric_col_keywords = ["ndcf", "cash flow", "profit", "amount", "cfo", "cfi", "cff", "pat"]
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in numeric_col_keywords):
                if "date" not in col_lower and "__raw" not in col:
                    df[col] = df[col].apply(_to_number)
        
        # Convert text columns
        text_cols = ["Entity", "Name of REIT", "Financial Year", "Period Ended"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(_strip)
        
        # Parse dates
        date_cols_to_parse = [c for c in df.columns if "date" in c.lower() and "__raw" not in c]
        for col in date_cols_to_parse:
            raw_col = f"{col}__raw"
            if raw_col in df.columns:
                df[col] = df[raw_col].apply(_to_date)
        
        # Filter out completely empty rows
        df = df.dropna(how='all')
        
        # Validate
        required = ["Financial Year", CONFIG.COMP_COL]
        is_valid, errors = validate_dataframe(df, required, "Trust sheet")
        if not is_valid:
            logger.warning(f"Trust data validation issues: {errors}")
        
        logger.info(f"Loaded {len(df)} rows from trust sheet '{sheet_name}'")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load trust sheet: {e}", exc_info=True)
        st.error(f"Error loading trust sheet: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_reit_spv_ndcf(url: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Load and process SPV/HoldCo NDCF data from Google Sheets.
    
    Args:
        url: Google Sheets URL
        sheet_name: Name of the sheet to load
        
    Returns:
        Processed DataFrame with numeric conversions
    """
    if sheet_name is None:
        sheet_name = CONFIG.SPV_SHEET_NAME
    
    try:
        csv_url = _csv_url_from_gsheet(url, sheet=sheet_name)
        df = pd.read_csv(csv_url, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        
        # Rename columns for consistency
        rename_map = {
            "Entity": "Name of REIT",
            "Fincial Year": "Financial Year",
            "Period": "Period Ended",
            "Period ended": "Period Ended",
            "Name of Holdco": "Name of Holdco (Leave Blank if N/A)",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Convert numeric columns
        numeric_col_keywords = ["ndcf", "cash flow", "profit", "spv", "holdco", "cfo", "cfi", "cff", "pat"]
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in numeric_col_keywords):
                df[col] = df[col].apply(_to_number)
        
        # Convert text columns
        for col in CONFIG.spv_text_cols:
            if col in df.columns:
                df[col] = df[col].apply(_strip)
        
        # Filter out completely empty rows
        df = df.dropna(how='all')
        
        # Validate
        required = ["Name of REIT", "Financial Year", "Name of SPV"]
        is_valid, errors = validate_dataframe(df, required, "SPV sheet")
        if not is_valid:
            logger.warning(f"SPV data validation issues: {errors}")
        
        logger.info(f"Loaded {len(df)} rows from SPV sheet '{sheet_name}'")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load SPV sheet: {e}", exc_info=True)
        st.error(f"Error loading SPV sheet: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_offer_doc_links(dir_url: Optional[str]) -> pd.DataFrame:
    """
    Load offer document links from directory workbook.
    
    Args:
        dir_url: URL to directory workbook containing links in Sheet5
        
    Returns:
        DataFrame with REIT names and OD links
    """
    if not dir_url:
        return pd.DataFrame(columns=["Name of REIT", "OD Link"])
    
    try:
        csv_url = _csv_url_from_gsheet(dir_url, sheet="Sheet5")
        df = pd.read_csv(csv_url, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        
        # Find entity and link columns
        ent_col = next((c for c in df.columns if "name" in c.lower() and ("reit" in c.lower() or "entity" in c.lower())), 
                       df.columns[0] if len(df.columns) > 0 else None)
        link_col = next((c for c in df.columns if "link" in c.lower()), 
                        df.columns[-1] if len(df.columns) > 1 else None)
        
        if ent_col and link_col:
            result = df[[ent_col, link_col]].rename(
                columns={ent_col: "Name of REIT", link_col: "OD Link"}
            )
            logger.info(f"Loaded {len(result)} offer document links")
            return result
        else:
            logger.warning("Could not identify columns in offer doc sheet")
            return pd.DataFrame(columns=["Name of REIT", "OD Link"])
            
    except Exception as e:
        logger.warning(f"Failed to load offer doc links from {dir_url}: {e}")
        return pd.DataFrame(columns=["Name of REIT", "OD Link"])

# -------- Computation Functions --------
def compute_payout_ratio(computed: pd.Series, declared: pd.Series, 
                        threshold: float = None) -> Tuple[pd.Series, pd.Series]:
    """
    Compute payout ratio and check against threshold.
    
    Args:
        computed: Series of computed NDCF values
        declared: Series of declared NDCF values
        threshold: Minimum required payout percentage (default: 90%)
        
    Returns:
        Tuple of (payout_ratio_series, meets_threshold_series)
    """
    if threshold is None:
        threshold = CONFIG.PAYOUT_THRESHOLD
    
    ratio = np.where(computed > 0, (declared / computed) * 100.0, np.nan)
    meets_threshold = pd.Series(ratio >= threshold)
    
    return pd.Series(ratio).round(2), meets_threshold

def compute_trust_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute compliance checks for trust-level NDCF data.
    
    Args:
        df: DataFrame containing trust-level NDCF data
        
    Returns:
        DataFrame with additional computed check columns
        
    Raises:
        ValueError: If required columns are missing
    """
    # Check for required columns with flexible matching
    comp_col = CONFIG.COMP_COL
    decl_col = CONFIG.DECL_INCL_COL
    cfo_col = CONFIG.CFO_COL
    cfi_col = CONFIG.CFI_COL
    cff_col = CONFIG.CFF_COL
    pat_col = CONFIG.PAT_COL
    
    # Find columns by keyword matching if exact match fails
    if comp_col not in df.columns:
        comp_col = next((c for c in df.columns if "ndcf computed" in c.lower()), None)
    if decl_col not in df.columns:
        decl_col = next((c for c in df.columns if "ndcf declared" in c.lower() and "incl" in c.lower()), None)
    if cfo_col not in df.columns:
        cfo_col = next((c for c in df.columns if "cash flow" in c.lower() and "operating" in c.lower()), None)
    if cfi_col not in df.columns:
        cfi_col = next((c for c in df.columns if "cash flow" in c.lower() and "investing" in c.lower()), None)
    if cff_col not in df.columns:
        cff_col = next((c for c in df.columns if "cash flow" in c.lower() and "financ" in c.lower()), None)
    if pat_col not in df.columns:
        pat_col = next((c for c in df.columns if "profit after tax" in c.lower()), None)
    
    missing = [c for c in [comp_col, decl_col, cfo_col, cfi_col, cff_col, pat_col] if c is None or c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for trust checks. Available columns: {df.columns.tolist()}")
    
    out = df.copy()
    
    # Check 1: Payout ratio
    out["Payout Ratio %"], out["Meets 90% Rule"] = compute_payout_ratio(
        out[comp_col], out[decl_col]
    )
    
    # Check 2: Cash flow reconciliation
    out["CF Sum"] = (
        out[cfo_col].fillna(0) + 
        out[cfi_col].fillna(0) + 
        out[cff_col].fillna(0) + 
        out[pat_col].fillna(0)
    )
    out["Gap vs Computed"] = out["CF Sum"] - out[comp_col]
    out["Gap % of Computed"] = np.where(
        out[comp_col] != 0,
        (out["Gap vs Computed"] / out[comp_col]) * 100.0,
        np.nan
    )
    out["Gap % of Computed"] = out["Gap % of Computed"].round(2)
    out["Within 10% Gap"] = out["Gap % of Computed"].abs() <= CONFIG.GAP_THRESHOLD
    
    # Store column names for display
    out["_comp_col"] = comp_col
    out["_decl_col"] = decl_col
    out["_cfo_col"] = cfo_col
    out["_cfi_col"] = cfi_col
    out["_cff_col"] = cff_col
    out["_pat_col"] = pat_col
    
    return out

def compute_trust_timeline_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute timeline compliance checks for trust-level data.
    
    Args:
        df: DataFrame containing date columns
        
    Returns:
        DataFrame with timeline check results
    """
    # Find date columns
    decl_col = next((c for c in df.columns if "declaration" in c.lower() and "date" in c.lower() and "__raw" not in c), None)
    rec_col = next((c for c in df.columns if "record" in c.lower() and "date" in c.lower() and "__raw" not in c), None)
    dist_col = next((c for c in df.columns if "distribution" in c.lower() and "date" in c.lower() and "__raw" not in c), None)
    
    if not all([decl_col, rec_col, dist_col]):
        logger.warning(f"Missing date columns for timeline checks. Found: decl={decl_col}, rec={rec_col}, dist={dist_col}")
        return pd.DataFrame(columns=[
            "Financial Year", "Period Ended", "Declaration Date", "Record Date", 
            "Distribution Date", "Days Decl→Record", "Record ≤ 2 days",
            "Days Record→Distr", "Distribution ≤ 5 days"
        ])
    
    t = df.copy()
    
    # Calculate day differences
    t["Days Decl→Record"] = (t[rec_col] - t[decl_col]).dt.days
    t["Days Record→Distr"] = (t[dist_col] - t[rec_col]).dt.days
    
    # Check compliance
    t["Record ≤ 2 days"] = (
        (t["Days Decl→Record"] >= 0) & 
        (t["Days Decl→Record"] <= CONFIG.RECORD_DATE_DAYS)
    )
    t["Distribution ≤ 5 days"] = (
        (t["Days Record→Distr"] >= 0) & 
        (t["Days Record→Distr"] <= CONFIG.DISTRIBUTION_DATE_DAYS)
    )
    
    # Rename for display
    t = t.rename(columns={
        decl_col: "Declaration Date",
        rec_col: "Record Date",
        dist_col: "Distribution Date"
    })
    
    return t[[
        "Financial Year", "Period Ended", "Declaration Date", "Record Date", 
        "Distribution Date", "Days Decl→Record", "Record ≤ 2 days",
        "Days Record→Distr", "Distribution ≤ 5 days"
    ]].copy()

def compute_spv_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute compliance checks for SPV/HoldCo level NDCF data.
    
    Args:
        df: DataFrame containing SPV-level NDCF data
        
    Returns:
        DataFrame with additional computed check columns
        
    Raises:
        ValueError: If required columns are missing
    """
    # Find columns flexibly
    comp_col = next((c for c in df.columns if "ndcf computed" in c.lower()), None)
    decl_col = next((c for c in df.columns if "ndcf declared" in c.lower() and "incl" in c.lower()), None)
    
    spv_cols = {
        'cfo': next((c for c in df.columns if "spv" in c.lower() and "operating" in c.lower()), None),
        'cfi': next((c for c in df.columns if "spv" in c.lower() and "investing" in c.lower()), None),
        'cff': next((c for c in df.columns if "spv" in c.lower() and "financ" in c.lower()), None),
        'pat': next((c for c in df.columns if "spv" in c.lower() and "profit" in c.lower()), None),
    }
    
    hco_cols = {
        'cfo': next((c for c in df.columns if "holdco" in c.lower() and "operating" in c.lower()), None),
        'cfi': next((c for c in df.columns if "holdco" in c.lower() and "investing" in c.lower()), None),
        'cff': next((c for c in df.columns if "holdco" in c.lower() and "financ" in c.lower()), None),
        'pat': next((c for c in df.columns if "holdco" in c.lower() and "profit" in c.lower()), None),
    }
    
    if not comp_col or not decl_col:
        raise ValueError(f"Missing required columns for SPV checks. Available columns: {df.columns.tolist()}")
    
    out = df.copy()
    
    # Check 1: Payout ratio
    out["Payout Ratio %"], out["Meets 90% Rule (SPV)"] = compute_payout_ratio(
        out[comp_col], out[decl_col]
    )
    
    # Check 2: SPV + HoldCo cash flow reconciliation
    cf_sum = 0
    for col in list(spv_cols.values()) + list(hco_cols.values()):
        if col and col in out.columns:
            cf_sum += out[col].fillna(0)
    
    out["SPV+HoldCo CF Sum"] = cf_sum
    out["Gap vs Computed (SPV)"] = out["SPV+HoldCo CF Sum"] - out[comp_col]
    out["Gap % of Computed (SPV)"] = np.where(
        out[comp_col] != 0,
        (out["Gap vs Computed (SPV)"] / out[comp_col]) * 100.0,
        np.nan
    ).round(2)
    out["Within Computed Bound (SPV)"] = np.where(
        out[comp_col] > 0,
        out["Gap vs Computed (SPV)"].abs() < out[comp_col],
        np.nan
    )
    
    # Store column names
    out["_comp_col"] = comp_col
    out["_decl_col"] = decl_col
    
    return out

# -------- UI Rendering Functions --------
def render_trust_level_analysis(df_trust: pd.DataFrame, entity: str, fy: str):
    """Render trust-level analysis section."""
    # Use Entity or Name of REIT
    entity_col = "Entity" if "Entity" in df_trust.columns else "Name of REIT"
    
    q = df_trust[(df_trust[entity_col] == entity) & 
                 (df_trust["Financial Year"] == fy)].copy()
    
    if q.empty:
        st.warning("No TRUST-level rows for the selected REIT and Financial Year.")
        return
    
    try:
        qc = compute_trust_checks(q)
    except ValueError as e:
        st.error(f"Cannot compute trust checks: {e}")
        return
    
    total = len(qc)
    meets_90 = int(qc['Meets 90% Rule'].fillna(False).sum())
    within_gap = int(qc['Within 10% Gap'].fillna(False).sum())
    
    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Periods meeting 90% payout", f"{meets_90}/{total}", 
              delta="Pass" if meets_90 == total else "Fail",
              delta_color="normal" if meets_90 == total else "inverse")
    c2.metric("Periods within 10% gap", f"{within_gap}/{total}",
              delta="Pass" if within_gap == total else "Fail",
              delta_color="normal" if within_gap == total else "inverse")
    c3.metric("Rows analysed", f"{total}")
    
    # Get column names
    comp_col = qc["_comp_col"].iloc[0] if "_comp_col" in qc.columns else CONFIG.COMP_COL
    decl_col = qc["_decl_col"].iloc[0] if "_decl_col" in qc.columns else CONFIG.DECL_INCL_COL
    cfo_col = qc["_cfo_col"].iloc[0] if "_cfo_col" in qc.columns else CONFIG.CFO_COL
    cfi_col = qc["_cfi_col"].iloc[0] if "_cfi_col" in qc.columns else CONFIG.CFI_COL
    cff_col = qc["_cff_col"].iloc[0] if "_cff_col" in qc.columns else CONFIG.CFF_COL
    pat_col = qc["_pat_col"].iloc[0] if "_pat_col" in qc.columns else CONFIG.PAT_COL
    
    # Check 1: Payout ratio
    st.subheader("Trust Check 1 – 90% payout of Computed NDCF")
    disp1 = qc[[
        "Financial Year", "Period Ended",
        comp_col, decl_col,
        "Payout Ratio %", "Meets 90% Rule",
    ]].copy()
    disp1 = disp1.rename(columns={
        comp_col: "Computed NDCF",
        decl_col: "Declared (incl. Surplus)"
    })
    disp1["Meets 90% Rule"] = disp1["Meets 90% Rule"].apply(_status)
    st.dataframe(disp1, use_container_width=True, hide_index=True)
    
    if (~qc["Meets 90% Rule"].fillna(False)).any():
        st.error("⚠️ TRUST: One or more periods do not meet 90% payout requirement.")
    else:
        st.success("✅ TRUST: All periods meet 90% payout requirement.")
    
    # Check 2: Cash flow reconciliation
    st.subheader("Trust Check 2 – (CFO + CFI + CFF + PAT) vs Computed NDCF")
    disp2 = qc[[
        "Financial Year", "Period Ended",
        cfo_col, cfi_col, cff_col, pat_col,
        "CF Sum", comp_col, "Gap vs Computed", "Gap % of Computed", "Within 10% Gap"
    ]].copy()
    disp2 = disp2.rename(columns={
        cfo_col: "CFO",
        cfi_col: "CFI",
        cff_col: "CFF",
        pat_col: "PAT",
        comp_col: "Computed NDCF"
    })
    disp2["Within 10% Gap"] = disp2["Within 10% Gap"].apply(_status)
    st.dataframe(disp2, use_container_width=True, hide_index=True)
    
    if (~qc["Within 10% Gap"].fillna(False)).any():
        st.error("⚠️ TRUST: One or more periods have a gap > 10% between cash flows and Computed NDCF.")
    else:
        st.success("✅ TRUST: All periods within 10% gap threshold.")
    
    # Timeline checks
    tline = compute_trust_timeline_checks(q)
    
    if not tline.empty:
        st.subheader("Trust Check 3a – Declaration → Record Date (≤ 2 days)")
        t1 = tline[[
            "Financial Year", "Period Ended", "Declaration Date", "Record Date",
            "Days Decl→Record", "Record ≤ 2 days"
        ]].copy()
        t1["Record ≤ 2 days"] = t1["Record ≤ 2 days"].apply(_status)
        st.dataframe(t1, use_container_width=True, hide_index=True)
        
        if (tline["Record ≤ 2 days"] == False).any():
            st.error("⚠️ TRUST: One or more periods have Record Date more than 2 days after Declaration.")
        else:
            st.success("✅ TRUST: All periods meet Record Date requirement (≤ 2 days).")
        
        st.subheader("Trust Check 3b – Record Date → Distribution Date (≤ 5 days)")
        t2 = tline[[
            "Financial Year", "Period Ended", "Record Date", "Distribution Date",
            "Days Record→Distr", "Distribution ≤ 5 days"
        ]].copy()
        t2["Distribution ≤ 5 days"] = t2["Distribution ≤ 5 days"].apply(_status)
        st.dataframe(t2, use_container_width=True, hide_index=True)
        
        if (tline["Distribution ≤ 5 days"] == False).any():
            st.error("⚠️ TRUST: One or more periods have Distribution Date more than 5 days after Record Date.")
        else:
            st.success("✅ TRUST: All periods meet Distribution Date requirement (≤ 5 days).")
    
    # Show raw date strings for debugging
    with st.expander("🔍 Show RAW date strings from sheet (for debugging)"):
        raw_cols = [c for c in q.columns if "__raw" in c]
        if raw_cols:
            display_cols = ["Financial Year", "Period Ended"] + raw_cols
            st.dataframe(q[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No raw date columns available")

def render_spv_level_analysis(df_spv: pd.DataFrame, entity: str, fy: str):
    """Render SPV/HoldCo level analysis section."""
    entity_col = "Entity" if "Entity" in df_spv.columns else "Name of REIT"
    
    q = df_spv[(df_spv[entity_col] == entity) & 
               (df_spv["Financial Year"] == fy)].copy()
    
    if q.empty:
        st.warning("No SPV rows for the selected REIT and Financial Year.")
        return
    
    try:
        qs = compute_spv_checks(q)
    except ValueError as e:
        st.error(f"Cannot compute SPV checks: {e}")
        return
    
    total = len(qs)
    meets_90_spv = int(qs['Meets 90% Rule (SPV)'].fillna(False).sum())
    within_bound = int(qs['Within Computed Bound (SPV)'].fillna(False).sum())
    
    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("SPV periods meeting 90% payout", f"{meets_90_spv}/{total}",
              delta="Pass" if meets_90_spv == total else "Fail",
              delta_color="normal" if meets_90_spv == total else "inverse")
    c2.metric("SPV periods within computed bound", f"{within_bound}/{total}",
              delta="Pass" if within_bound == total else "Fail",
              delta_color="normal" if within_bound == total else "inverse")
    c3.metric("SPV rows analysed", f"{total}")
    
    # Get column names
    comp_col = qs["_comp_col"].iloc[0] if "_comp_col" in qs.columns else CONFIG.COMP_COL
    decl_col = qs["_decl_col"].iloc[0] if "_decl_col" in qs.columns else CONFIG.DECL_INCL_COL
    
    # Check 1: SPV Payout ratio
    st.subheader("SPV Check 1 – Declared (incl. Surplus) ≥ 90% of Computed (by SPV/period)")
    spv_name_col = "Name of SPV" if "Name of SPV" in qs.columns else next((c for c in qs.columns if "spv" in c.lower()), "SPV")
    holdco_name_col = "Name of Holdco (Leave Blank if N/A)" if "Name of Holdco (Leave Blank if N/A)" in qs.columns else next((c for c in qs.columns if "holdco" in c.lower()), "HoldCo")
    
    d1 = qs[[
        spv_name_col, holdco_name_col, "Financial Year", "Period Ended",
        comp_col, decl_col, "Payout Ratio %", "Meets 90% Rule (SPV)"
    ]].copy()
    d1 = d1.rename(columns={
        comp_col: "Computed NDCF",
        decl_col: "Declared (incl. Surplus)",
        spv_name_col: "SPV Name",
        holdco_name_col: "HoldCo Name"
    })
    d1["Meets 90% Rule (SPV)"] = d1["Meets 90% Rule (SPV)"].apply(_status)
    st.dataframe(d1, use_container_width=True, hide_index=True)
    
    if (~qs["Meets 90% Rule (SPV)"].fillna(False)).any():
        st.error("⚠️ SPV: One or more SPV periods do not meet 90% payout requirement.")
    else:
        st.success("✅ SPV: All SPV periods meet 90% payout requirement.")
    
    # Check 2: SPV + HoldCo reconciliation
    st.subheader("SPV Check 2 – |(SPV+HoldCo CFO+CFI+CFF+PAT) − Computed| < Computed")
    d2 = qs[[
        spv_name_col, holdco_name_col, "Financial Year", "Period Ended",
        "SPV+HoldCo CF Sum", comp_col, "Gap vs Computed (SPV)", 
        "Gap % of Computed (SPV)", "Within Computed Bound (SPV)"
    ]].copy()
    d2 = d2.rename(columns={
        comp_col: "Computed NDCF",
        spv_name_col: "SPV Name",
        holdco_name_col: "HoldCo Name"
    })
    d2["Within Computed Bound (SPV)"] = d2["Within Computed Bound (SPV)"].apply(_status)
    st.dataframe(d2, use_container_width=True, hide_index=True)
    
    if (~qs["Within Computed Bound (SPV)"].fillna(False)).any():
        st.error("⚠️ SPV: One or more SPV periods have |Gap| ≥ Computed NDCF.")
    else:
        st.success("✅ SPV: All SPV periods within computed bound.")

def render():
    """Main render function for NDCF compliance checks."""
    st.header("NDCF – Compliance Checks")
    
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        
        seg = st.selectbox(
            "Select Segment", 
            ["REIT", "InvIT"], 
            index=0,
            help="Choose between REIT or InvIT for compliance checks"
        )
        
        data_url = st.text_input(
            "Data URL (Google Sheet - public view)",
            value=CONFIG.DEFAULT_SHEET_URL_TRUST,
            help=f"Trust sheet: '{CONFIG.TRUST_SHEET_NAME}'. SPV sheet: '{CONFIG.SPV_SHEET_NAME}'.",
        )
        
        if st.button("🔄 Refresh Data", help="Clear cache and reload data"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.caption("📊 NDCF Compliance Analyzer v2.0")
    
    if seg != "REIT":
        st.info("📝 InvIT compliance checks will be added in a future update.")
        return
    
    # Load data with progress indicator
    with st.spinner("Loading REIT data..."):
        df_trust_all = load_reit_ndcf(data_url, CONFIG.TRUST_SHEET_NAME)
        df_spv_all = load_reit_spv_ndcf(data_url, CONFIG.SPV_SHEET_NAME)
    
    if df_trust_all.empty:
        st.error("❌ Trust sheet appears empty or could not be loaded. Please check the URL and sheet name.")
        st.info("💡 Make sure the Google Sheet is publicly accessible (Anyone with the link can view)")
        return
    
    # Entity selection - use Entity or Name of REIT column
    entity_col = "Entity" if "Entity" in df_trust_all.columns else "Name of REIT"
    available_entities = sorted(df_trust_all[entity_col].dropna().unique().tolist())
    
    if not available_entities:
        st.error("❌ No REITs found in the data.")
        return
    
    ent = st.selectbox(
        "Choose REIT",
        available_entities,
        index=0,
        key="ndcf_reit_select",
        help="Select the REIT to analyze"
    )
    
    # Display Offer Document link if available
    if CONFIG.DEFAULT_REIT_DIR_URL:
        with st.spinner("Loading offer document links..."):
            od_df = load_offer_doc_links(CONFIG.DEFAULT_REIT_DIR_URL)
        
        if not od_df.empty:
            od_links = od_df.loc[od_df["Name of REIT"] == ent, "OD Link"]
            if not od_links.empty and od_links.iloc[0].strip():
                st.markdown(f"📄 **Offer Document:** [{od_links.iloc[0].strip()}]({od_links.iloc[0].strip()})")
    
    # Analysis level selection
    level = st.radio(
        "Analysis level", 
        ["Trust", "SPV/HoldCo"], 
        horizontal=True, 
        key="ndcf_level_select",
        help="Choose between Trust-level or SPV/HoldCo-level analysis"
    )
    
    # Financial Year selection based on analysis level
    if level == "Trust":
        fy_options = sorted(
            df_trust_all.loc[df_trust_all[entity_col] == ent, "Financial Year"]
            .dropna().unique().tolist()
        )
    else:
        if df_spv_all.empty:
            st.info("📝 SPV sheet could not be loaded. Only Trust-level analysis is available.")
            return
        spv_entity_col = "Entity" if "Entity" in df_spv_all.columns else "Name of REIT"
        fy_options = sorted(
            df_spv_all.loc[df_spv_all[spv_entity_col] == ent, "Financial Year"]
            .dropna().unique().tolist()
        )
    
    if not fy_options:
        st.warning(f"⚠️ No financial years found for {ent} at {level} level.")
        return
    
    fy = st.selectbox(
        "Financial Year", 
        ["– Select –"] + fy_options, 
        index=0, 
        key="ndcf_fy_select",
        help="Select the financial year to analyze"
    )
    
    if fy == "– Select –":
        st.info("👆 Please select a Financial Year to view compliance results.")
        return
    
    # Render analysis based on selected level
    st.divider()
    
    if level == "Trust":
        render_trust_level_analysis(df_trust_all, ent, fy)
    else:
        render_spv_level_analysis(df_spv_all, ent, fy)

def render_ndcf():
    """Wrapper function for compatibility."""
    render()

if __name__ == "__main__":
    render()