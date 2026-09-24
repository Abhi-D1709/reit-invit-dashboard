"""Cross-entity (peer / sector) benchmarking built from the NSE master filing
list. The base metrics (sponsor %, public %) come for free from the master
list already loaded for the sidebar; an optional extended pass fetches and
parses each entity's latest XBRL filing to add domestic/foreign and lock-in
metrics.
"""

import pandas as pd
import streamlit as st

from utils.uhp import sources as filing_source
from utils.uhp import ownership_reports
from utils.uhp.xbrl_parser import parse_uhp_xbrl, to_number


def latest_filing_per_entity(master_df: pd.DataFrame) -> pd.DataFrame:
    idx = master_df.groupby("entityKey")["asOnDateParsed"].idxmax()
    return master_df.loc[idx].sort_values("publicHoldingPer", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner="Fetching and parsing the latest filing for every entity...")
def build_extended_peer_metrics(latest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in latest_df.iterrows():
        try:
            xml_text = filing_source.fetch_xbrl(rec["xbrlFilePath"])
            parsed = parse_uhp_xbrl(xml_text)
        except Exception:
            continue
        dom_for = ownership_reports.build_domestic_foreign_report(parsed)["table"]
        overall = dom_for[dom_for["Segment"] == "Overall"].iloc[0]
        locked_in_pct = to_number(
            parsed.get("TotalUnitsOutstandingI", "AsAPercentageOfTotalUnitsMandatorilyHeld")
        )
        pledged_pct = to_number(
            parsed.get("TotalUnitsOutstandingI", "AsAPercentageOfTotalNumberOfUnitsPledgedOrOtherwiseEncumbered")
        )
        rows.append(
            {
                "entityKey": rec["entityKey"],
                "Foreign holding %": overall["Foreign %"],
                "Domestic holding %": overall["Domestic %"],
                "Units locked-in %": locked_in_pct,
                "Units pledged %": pledged_pct,
            }
        )
    return pd.DataFrame(rows)
