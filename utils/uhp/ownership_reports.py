"""Derived ownership analysis reports built on top of the SEBI Unit Holding
Pattern XBRL data — views the raw taxonomy doesn't give directly (e.g.
domestic vs. foreign ownership), assembled from the underlying categories.
"""

import re

import pandas as pd

from utils.uhp.xbrl_parser import ParsedUHP, to_number

FOREIGN_KEYWORDS = re.compile(
    r"foreign|overseas|non.?resident|\bnri\b|\bfpi\b|\bfii\b|\bfvci\b", re.IGNORECASE
)

# Sponsor Group: the taxonomy already splits this into Indian vs Foreign explicitly.
SPONSOR_DOMESTIC_CONTEXT = "IndianI"
SPONSOR_FOREIGN_CONTEXT = "ForeignI"
SPONSOR_TOTAL_CONTEXT = "UnitHoldingOfSponsorAndSponsorGroupI"

# Public: leaf categories that are foreign / domestic by definition.
PUBLIC_FOREIGN_LEAF_CONTEXTS = [
    ("ForeignPortfolioInvestorsInstitutionsI", "Foreign Portfolio Investors"),
    ("ForeignVentureCapitalInvestorsInstitutionsI", "Foreign Venture Capital Investors"),
    ("NonResidentIndiansI", "Non-Resident Indians (NRIs)"),
]
PUBLIC_DOMESTIC_LEAF_CONTEXTS = [
    ("MutualFundsInstitutionsI", "Mutual Funds"),
    ("FinancialInstitutionsOrBanksInstitutionsI", "Financial Institutions/Banks"),
    ("CentralGovernmentOrStateGovernmentsI", "Central Government/State Government(s)"),
    ("VentureCapitalFundsInstitutionsI", "Venture Capital Funds"),
    ("InsuranceCompaniesInstitutionsI", "Insurance Companies"),
    ("ProvidentOrPensionFundsInstitutionsI", "Provident Funds/Pension Funds"),
    ("CentralGovernmentOrStateGovernmentsOrPresidentOfIndiaNonInstitutionsI", "Central/State Govt(s)/President of India"),
    ("IndividualsNonInstitutionsI", "Individuals (resident)"),
    ("NBFCsRegisteredWithRBINonInstitutionsI", "NBFCs registered with RBI"),
    ("TrustsI", "Trusts"),
    ("ClearingsI", "Clearing Members"),
    ("BodyCorporatesI", "Bodies Corporate"),
]

# "Any Other" aggregate buckets under Public that need break-up classification.
PUBLIC_ANY_OTHER = [
    ("OtherInstitutions", "OtherInstitutionsI", "Any Other — Institutions"),
    ("OtherNonInstitutions", "OtherNonInstitutionsI", "Any Other — Non-Institutions"),
]


def _units(parsed: ParsedUHP, cid: str) -> float:
    return to_number(parsed.facts.get(cid, {}).get("NumberOfUnitsHeld"))


def _classify_any_other(parsed: ParsedUHP, prefix: str, aggregate_cid: str):
    """Splits an 'Any Other' bucket into domestic/foreign/unclassified units
    using keyword matching on each break-up entity's disclosed nature, and
    returns the per-entity audit trail alongside the totals.
    """
    domestic = 0.0
    foreign = 0.0
    classified_total = 0.0
    audit_rows = []
    n = 1
    while True:
        cid = f"{prefix}{n}I"
        if cid not in parsed.facts:
            break
        f = parsed.facts[cid]
        units = to_number(f.get("NumberOfUnitsHeld"))
        nature = f.get("NatureOfOther") or "(unspecified)"
        is_foreign = bool(FOREIGN_KEYWORDS.search(nature))
        if is_foreign:
            foreign += units
        else:
            domestic += units
        classified_total += units
        audit_rows.append(
            {
                "Nature of 'Any Other'": nature,
                "No. of units held": units,
                "Classified as": "Foreign" if is_foreign else "Domestic",
            }
        )
        n += 1

    aggregate_units = _units(parsed, aggregate_cid)
    if n == 1:
        # No break-up disclosed at all — the lump sum can't be classified.
        return 0.0, 0.0, aggregate_units, audit_rows

    residual = aggregate_units - classified_total
    unclassified = residual if abs(residual) > 0.5 else 0.0
    return domestic, foreign, max(unclassified, 0.0), audit_rows


def build_domestic_foreign_report(parsed: ParsedUHP) -> dict:
    total_units = _units(parsed, "TotalUnitsOutstandingI")

    sponsor_domestic = _units(parsed, SPONSOR_DOMESTIC_CONTEXT)
    sponsor_foreign = _units(parsed, SPONSOR_FOREIGN_CONTEXT)
    sponsor_total = _units(parsed, SPONSOR_TOTAL_CONTEXT)
    sponsor_unclassified = max(sponsor_total - sponsor_domestic - sponsor_foreign, 0.0)

    public_domestic = sum(_units(parsed, c) for c, _ in PUBLIC_DOMESTIC_LEAF_CONTEXTS)
    public_foreign = sum(_units(parsed, c) for c, _ in PUBLIC_FOREIGN_LEAF_CONTEXTS)
    public_unclassified = 0.0
    audit_rows: list[dict] = []
    for prefix, agg_cid, bucket_label in PUBLIC_ANY_OTHER:
        d, f, u, rows = _classify_any_other(parsed, prefix, agg_cid)
        public_domestic += d
        public_foreign += f
        public_unclassified += u
        for r in rows:
            r["Bucket"] = bucket_label
            audit_rows.append(r)

    public_total = _units(parsed, "PublicHoldingI")
    reconciled = public_domestic + public_foreign + public_unclassified
    drift = public_total - reconciled
    if abs(drift) > 0.5:
        public_unclassified += drift

    overall_domestic = sponsor_domestic + public_domestic
    overall_foreign = sponsor_foreign + public_foreign
    overall_unclassified = sponsor_unclassified + public_unclassified

    def pct(x):
        return (x / total_units * 100) if total_units else 0.0

    def make_row(segment, domestic, foreign, unclassified, total):
        return {
            "Segment": segment,
            "Domestic units": domestic,
            "Domestic %": pct(domestic),
            "Foreign units": foreign,
            "Foreign %": pct(foreign),
            "Unclassified units": unclassified,
            "Unclassified %": pct(unclassified),
            "Total units": total,
        }

    table = pd.DataFrame(
        [
            make_row("Sponsor & Sponsor Group", sponsor_domestic, sponsor_foreign, sponsor_unclassified, sponsor_total),
            make_row("Public", public_domestic, public_foreign, public_unclassified, public_total),
            make_row("Overall", overall_domestic, overall_foreign, overall_unclassified, total_units),
        ]
    )

    audit_df = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame()

    return {
        "table": table,
        "audit": audit_df,
        "has_unclassified": overall_unclassified > 0.5,
    }
