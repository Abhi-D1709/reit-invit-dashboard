"""Encodes SEBI's prescribed Unit Holding Pattern format for REITs/InvITs
and builds pandas DataFrames from a ParsedUHP instance.

Reference taxonomy: SEBI in-capmkt XBRL taxonomy for Unit Holding Pattern
(continuous disclosure) of REITs/InvITs, as filed with NSE.
"""

import numpy as np
import pandas as pd

from utils.uhp.xbrl_parser import ParsedUHP, to_number

NUMERIC_FIELDS = [
    ("NumberOfUnitsHeld", "No. of units held"),
    ("AsAPercentageOfTotalOutStandingUnits", "As a % of total outstanding units"),
    ("NumberOfUnitsMandatorilyHeld", "No. of units mandatorily / locked-in held"),
    ("AsAPercentageOfTotalUnitsMandatorilyHeld", "As a % of total units mandatorily held"),
    ("NumberOfUnitsPledgedOrOtherwiseEncumbered", "No. of units pledged/otherwise encumbered"),
    (
        "AsAPercentageOfTotalNumberOfUnitsPledgedOrOtherwiseEncumbered",
        "As a % of total units pledged/otherwise encumbered",
    ),
]

# (context_id or marker, indent level, label, row kind)
# row kind: None (section/subsection header, no data), "row", "subtotal", "total", "grand_total"
TABLE_I_SCHEMA = [
    ("__SECTION__", 0, "(A) Unit Holding of Sponsor and Sponsor Group", None),
    ("__SUBSECTION__", 1, "(1) Indian", None),
    ("IndividualsOrHUFI", 2, "Individuals/HUF", "row"),
    ("CentralGovernmentOrStateGovernmentI", 2, "Central Government/State Government(s)", "row"),
    ("FinancialInstitutionsBanksI", 2, "Financial Institutions/Banks", "row"),
    ("OtherIndianI", 2, "Any Other (specify)", "row"),
    ("IndianI", 1, "Sub-Total (A)(1)", "subtotal"),
    ("__SUBSECTION__", 1, "(2) Foreign", None),
    ("IndividualsNonResidentIndiansOrForeignIndividualsI", 2, "Individuals (NRIs/Foreign Individuals)", "row"),
    ("ForeignGovernmentI", 2, "Government", "row"),
    ("ForeignInstitutionsI", 2, "Institutions", "row"),
    ("ForeignPortfolioInvestorsI", 2, "Foreign Portfolio Investors", "row"),
    ("OtherForeignI", 2, "Any Other (specify)", "row"),
    ("ForeignI", 1, "Sub-Total (A)(2)", "subtotal"),
    (
        "UnitHoldingOfSponsorAndSponsorGroupI",
        0,
        "Total Unit Holding of Sponsor and Sponsor Group (A) = (A)(1)+(A)(2)",
        "total",
    ),
    ("__SECTION__", 0, "(B) Public Holding", None),
    ("__SUBSECTION__", 1, "(1) Institutions", None),
    ("MutualFundsInstitutionsI", 2, "Mutual Funds", "row"),
    ("FinancialInstitutionsOrBanksInstitutionsI", 2, "Financial Institutions/Banks", "row"),
    ("CentralGovernmentOrStateGovernmentsI", 2, "Central Government/State Government(s)", "row"),
    ("VentureCapitalFundsInstitutionsI", 2, "Venture Capital Funds", "row"),
    ("InsuranceCompaniesInstitutionsI", 2, "Insurance Companies", "row"),
    ("ProvidentOrPensionFundsInstitutionsI", 2, "Provident Funds/Pension Funds", "row"),
    ("ForeignPortfolioInvestorsInstitutionsI", 2, "Foreign Portfolio Investors", "row"),
    ("ForeignVentureCapitalInvestorsInstitutionsI", 2, "Foreign Venture Capital Investors", "row"),
    ("OtherInstitutionsI", 2, "Any Other (specify)", "row"),
    ("InstitutionsI", 1, "Sub-Total (B)(1)", "subtotal"),
    ("__SUBSECTION__", 1, "(2) Non-Institutions", None),
    (
        "CentralGovernmentOrStateGovernmentsOrPresidentOfIndiaNonInstitutionsI",
        2,
        "Central Government/State Government(s)/President of India",
        "row",
    ),
    ("IndividualsNonInstitutionsI", 2, "Individuals", "row"),
    ("NBFCsRegisteredWithRBINonInstitutionsI", 2, "NBFCs registered with RBI", "row"),
    ("TrustsI", 2, "Trusts", "row"),
    ("NonResidentIndiansI", 2, "Non-Resident Indians (NRIs)", "row"),
    ("ClearingsI", 2, "Clearing Members", "row"),
    ("BodyCorporatesI", 2, "Bodies Corporate", "row"),
    ("OtherNonInstitutionsI", 2, "Any Other (specify)", "row"),
    ("NonInstitutionsI", 1, "Sub-Total (B)(2)", "subtotal"),
    ("PublicHoldingI", 0, "Total Public Holding (B) = (B)(1)+(B)(2)", "total"),
    ("TotalUnitsOutstandingI", 0, "Total Unit Holding (A)+(B)", "grand_total"),
]

# "Any Other (specify)" break-up groups: context id prefix -> descriptive title
ANY_OTHER_GROUPS = [
    ("OtherIndian", "Break-up of 'Any Other' - Sponsor Group (Indian)"),
    ("OtherForeign", "Break-up of 'Any Other' - Sponsor Group (Foreign)"),
    ("OtherInstitutions", "Break-up of 'Any Other' - Public Institutions"),
    ("OtherNonInstitutions", "Break-up of 'Any Other' - Public Non-Institutions"),
]

HEADER_FIELD_MAP = [
    ("NameOfTheCompany", "Name of the Entity"),
    ("NSESymbol", "NSE Symbol"),
    ("ScripCode", "BSE Scrip Code"),
    ("SebiRegistrationNumber", "SEBI Registration Number"),
    ("TypeOfReportREITsINVITs", "Type of Report"),
    ("NumberOfSecurities", "Total No. of Units Outstanding"),
    ("DateOfStartOfFinancialYear", "Financial Year Start"),
    ("DateOfEndOfFinancialYear", "Financial Year End"),
    ("CapitalRestructuringDateOrListingDate", "Listing / Capital Restructuring Date"),
]


NOT_LISTED_PLACEHOLDERS = {"", "000000", "NA", "N/A", "NOTLISTED", "NOT LISTED", "NOT APPLICABLE", "NIL"}


def build_header_info(parsed: ParsedUHP) -> dict:
    facts = parsed.facts.get("OneI", {})
    info = {label: facts.get(key) for key, label in HEADER_FIELD_MAP}
    for label in ("BSE Scrip Code", "NSE Symbol"):
        if (info.get(label) or "").strip().upper() in NOT_LISTED_PLACEHOLDERS:
            info[label] = "Not listed"
    if not info.get("Total No. of Units Outstanding"):
        total = parsed.get("TotalUnitsOutstandingI", "NumberOfUnitsHeld")
        info["Total No. of Units Outstanding"] = total
    return info


def build_table_i(parsed: ParsedUHP) -> pd.DataFrame:
    rows = []
    for context_id, level, label, kind in TABLE_I_SCHEMA:
        if kind is None:
            row = {"Category": (" " * (level * 4)) + label, "_kind": "header"}
            for _, col_label in NUMERIC_FIELDS:
                row[col_label] = np.nan
            rows.append(row)
            continue

        f = parsed.facts.get(context_id, {})
        row = {"Category": (" " * (level * 4)) + label, "_kind": kind}
        for field_key, col_label in NUMERIC_FIELDS:
            row[col_label] = to_number(f.get(field_key)) if f else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_any_other_breakup(parsed: ParsedUHP) -> dict[str, pd.DataFrame]:
    tables = {}
    for prefix, title in ANY_OTHER_GROUPS:
        rows = []
        n = 1
        while True:
            cid = f"{prefix}{n}I"
            if cid not in parsed.facts:
                break
            f = parsed.facts[cid]
            row = {"S.No.": n, "Nature of 'Any Other'": f.get("NatureOfOther")}
            for field_key, col_label in NUMERIC_FIELDS:
                row[col_label] = to_number(f.get(field_key))
            rows.append(row)
            n += 1
        if rows:
            tables[title] = pd.DataFrame(rows)
    return tables


def build_other_unitholders_table(parsed: ParsedUHP) -> pd.DataFrame:
    rows = []
    n = 1
    while True:
        cid = f"UnitHoldersOtherThanSponsorOfTheREITINVIT{n}I"
        if cid not in parsed.facts:
            break
        f = parsed.facts[cid]
        rows.append(
            {
                "S.No.": n,
                "Name of Unit Holder": f.get("NameOfTheUnitHolder"),
                "Related to Sponsor/any Party to the REIT/InvIT": f.get("RelatedToAnyPartyToTheREITINVIT"),
                "% of Holding": round(to_number(f.get("PercentageOfHolding")) * 100, 4),
            }
        )
        n += 1
    return pd.DataFrame(rows)


def build_manager_shareholders_table(parsed: ParsedUHP) -> pd.DataFrame:
    rows = []
    n = 1
    while True:
        cid = f"ShareholdersOfTheManangerOrInvestmentManagerOfREITINVIT{n}I"
        if cid not in parsed.facts:
            break
        f = parsed.facts[cid]
        rows.append(
            {
                "S.No.": n,
                "Name of Shareholder/Partner": f.get("NameOfTheUnitHolderOrPartner"),
                "Related to Sponsor/any other Party to the REIT/InvIT": f.get(
                    "RelatedToSponsorOrAnyOtherPartyToTheREITINVIT"
                ),
                "% of Holding/Share": round(to_number(f.get("PercentageOfHoldingShare")) * 100, 4),
            }
        )
        n += 1
    return pd.DataFrame(rows)


def build_directors_kmp_table(parsed: ParsedUHP) -> pd.DataFrame:
    rows = []
    n = 1
    while True:
        name_f = parsed.facts.get(f"ITABLE{n:03d}I") or parsed.facts.get(f"ITABLEI{n}")
        if name_f is None:
            break
        detail_f = parsed.facts.get(f"ITABLEI{n:03d}I") or name_f
        rows.append(
            {
                "S.No.": n,
                "Name": name_f.get("NameOfDirectorsOrKMPsOfTheManagerOrIMOfREITINVIT"),
                "Designation": detail_f.get("DesignationOfDirectorsOrKMPsOfTheManagerOrIMOfREITINVIT"),
                "Appointment/Resignation/Removal": detail_f.get(
                    "AppointmentOrResignationOrRemovalOfDirectorsOrKMPsOfTheManagerOrIMOfREITINVIT"
                ),
                "Date": detail_f.get("DateOfAppointmentOrResignationOrRemovalOfDirectorsOrKMPsOfTheManagerOrIMOfREITINVIT"),
                "Brief Profile": detail_f.get(
                    "BriefProfileInCaseOfAppointmentOfDirectorsOrKMPsOfTheManagerOrIMOfREITINVIT"
                ),
            }
        )
        n += 1
    return pd.DataFrame(rows)


def build_category_breakdown(parsed: ParsedUHP) -> pd.DataFrame:
    """Flat table of every leaf category (for charting), excluding subtotal/total rows."""
    rows = []
    for context_id, level, label, kind in TABLE_I_SCHEMA:
        if kind != "row":
            continue
        f = parsed.facts.get(context_id, {})
        pct = to_number(f.get("AsAPercentageOfTotalOutStandingUnits"))
        units = to_number(f.get("NumberOfUnitsHeld"))
        if units or pct:
            rows.append({"Category": label, "No. of Units Held": units, "% of Total Units": pct})
    return pd.DataFrame(rows)
