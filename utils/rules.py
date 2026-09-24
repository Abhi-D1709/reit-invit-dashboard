# utils/rules.py
"""Every compliance threshold the dashboard applies, in one place.

Change a number here and every page follows. Each rule records where it comes from and how
far it has been checked, and the "Rules reference" page shows this table to users.

How to read `status`:
  VERIFIED    checked against the regulation text (see `source`)
  PARTIAL     the figure is in the regulation, but the way the app applies it was not fully checked
  HOUSE_RULE  a deliberate alert or heuristic chosen by the dashboard owner, not a regulatory figure
  UNVERIFIED  carried over from earlier code; not checked against a source

`enforced` is False for rules recorded here but not (yet) checked by any page.

Regulation text reviewed: SEBI (Real Estate Investment Trusts) Regulations, 2014, consolidated
version last amended 23 Oct 2023 (the "REIT Regulations"), and SEBI circular
SEBI/HO/DDHS/PoD2/P/CIR/2023/106 of 27 Jun 2023. Later amendments and the InvIT Regulations
have NOT been checked, except the distribution timeline of Reg. 18(16)(c), which the Third
Amendment Regulations 2024 changed from 27 Nov 2024 (effective date from SEBI circular
SEBI/HO/DDHS/DDHS-PoD-2/P/CIR/2024/158 of 13 Nov 2024; the 2 and 5 working-day figures come from
published summaries of the amendment, not from its text).

Use the constants as `rules.NAME` (attribute lookup at call time), never `from utils.rules import NAME`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

VERIFIED = "verified"
PARTIAL = "partly verified"
HOUSE_RULE = "house rule"
UNVERIFIED = "not verified"

REGS_CHECKED = "REIT Regulations 2014, consolidated, last amended 23 Oct 2023"
REGS_URL = "https://www.sebi.gov.in/sebi_data/attachdocs/jan-2024/1705989608182.pdf"


@dataclass(frozen=True)
class Rule:
    key: str
    area: str
    name: str
    value: object
    unit: str
    applies_to: str
    source: str
    status: str
    note: str = ""
    enforced: bool = True


RULES: dict[str, Rule] = {}


def _rule(key, area, name, value, unit, applies_to, source, status, note="", enforced=True):
    """Register a rule and return its value (so it can be assigned to a module constant)."""
    if key in RULES:
        raise ValueError(f"duplicate rule key {key!r}")
    RULES[key] = Rule(key, area, name, value, unit, applies_to, source, status, note, enforced)
    return value


# --------------------------------------------------------------- sponsor / public holding
SPONSOR_MIN_INITIAL = _rule(
    "sponsor.min_initial", "Sponsor & public holding", "Sponsor + sponsor group minimum unitholding, first years after listing",
    0.15, "fraction of total units", "REIT",
    "REIT Regulations, Reg. 11(3)(i) (as substituted w.e.f. 17 Aug 2023; the version before it also said 15% for 3 years)", VERIFIED,
    "Units above 15% must be held for at least 1 year from listing.",
)
SPONSOR_MIN_INITIAL_YEARS = _rule(
    "sponsor.min_initial_years", "Sponsor & public holding", "Years from listing for which the initial sponsor minimum applies",
    3.0, "years", "REIT", "REIT Regulations, Reg. 11(3)(i)", VERIFIED,
)
SPONSOR_GRADED_MINIMUMS = _rule(
    "sponsor.graded", "Sponsor & public holding", "Sponsor minimum after year 3: 5% (yrs 4-5), 3% (yrs 6-10), 2% (yrs 11-20), 1% (after 20)",
    ((4, 5, 0.05), (6, 10, 0.03), (11, 20, 0.02), (21, None, 0.01)), "fraction of total units by year range", "REIT",
    "REIT Regulations, Reg. 11(3)(ii)-(v)", VERIFIED,
    "Capped at Rs 500 crore of value on the latest NAV; assessed at each fresh issue and at each change of threshold; "
    "REITs listed before 17 Aug 2023 apply it only to units issued after that date. Needs NAV, which the sheets don't have.",
    enforced=False,
)
PUBLIC_MIN = _rule(
    "public.min", "Sponsor & public holding", "Minimum public unitholding",
    0.25, "fraction of total units", "REIT",
    "REIT Regulations, Reg. 14(2A) (second proviso); SEBI circular of 27 Jun 2023", VERIFIED,
    "A listed REIT below 25% must reach 25% within 3 years of listing. Larger REITs may list with less "
    "(Rs 400 crore, or 10%, depending on size).",
)
PUBLIC_MIN_DEADLINE_YEARS = _rule(
    "public.deadline_years", "Sponsor & public holding", "Years from listing to reach the minimum public unitholding",
    3.0, "years", "REIT", "REIT Regulations, Reg. 14(2A) (second proviso)", VERIFIED,
)

# --------------------------------------------------------------------------- borrowings
REIT_NBR_TRIGGER = _rule(
    "borrowings.reit_trigger", "Borrowings", "REIT: net borrowings above this need a credit rating and unitholder approval for further borrowing",
    0.25, "fraction of value of REIT assets", "REIT", "REIT Regulations, Reg. 20(3)", VERIFIED,
)
REIT_NBR_CAP = _rule(
    "borrowings.reit_cap", "Borrowings", "REIT: net borrowings must never exceed",
    0.49, "fraction of value of REIT assets", "REIT",
    "REIT Regulations, Reg. 20(2); a breach caused by market movements must be cured within 6 months (Reg. 20(4))", VERIFIED,
    "Alerted on the Borrowings page. Refundable tenant deposits are excluded from borrowings and cash from asset value; "
    "the 6-month cure period and the trustee intimation are not checked.",
)
INVIT_NBR_CAP = _rule(
    "borrowings.invit_cap", "Borrowings", "InvIT: net borrowings cap",
    0.70, "fraction of value of assets", "InvIT", "InvIT Regulations (not checked)", UNVERIFIED,
)
INVIT_AAA_THRESHOLD = _rule(
    "borrowings.invit_aaa", "Borrowings", "InvIT: above this, an AAA rating and unitholder approval are required",
    0.49, "fraction of value of assets", "InvIT", "InvIT Regulations (not checked)", UNVERIFIED,
)
INVIT_RATING_TRIGGER = _rule(
    "borrowings.invit_rating", "Borrowings", "InvIT: above this, a credit rating and unitholder approval are required",
    0.25, "fraction of value of assets", "InvIT", "InvIT Regulations (not checked)", UNVERIFIED,
)

# -------------------------------------------------------------------------- investments
INVEST_COMPLETED_MIN_PCT = _rule(
    "investment.completed_min", "Investments", "Minimum share of REIT assets in completed, rent and/or income generating properties",
    80.0, "% of value of REIT assets", "REIT", "REIT Regulations, Reg. 18(4)", VERIFIED,
)
INVEST_ALERT_BAND = _rule(
    "investment.alert_band", "Investments", "Extra alert band on the completed-assets ratio (inclusive)",
    (81.0, 85.0), "% of value of REIT assets", "REIT", "Dashboard owner's alert; not a regulatory figure", HOUSE_RULE,
    "Ratios inside this band raise a red alert even though they are above 80%.",
)
SPV_HOLDING_MAX_PCT = _rule(
    "investment.spv_holding_max", "Investments", "Alert when a shareholder holds more than this in an SPV that is not wholly owned",
    50.0, "% holding", "REIT", "Earlier dashboard rule (not checked)", UNVERIFIED,
)

# ------------------------------------------------------------------------------- NDCF
NDCF_PAYOUT_MIN_PCT = _rule(
    "ndcf.payout_min", "NDCF distribution", "Minimum distribution of net distributable cash flows",
    90.0, "% of NDCF", "REIT (trust and SPV level)",
    "REIT Regulations, Reg. 18(16)(a) (SPV to REIT) and 18(16)(b) (REIT to unitholders)", VERIFIED,
    "At holdco level 100% of cash received from SPVs must go to the REIT (Reg. 18(16)(aa)); not checked.",
)
NDCF_NEW_TIMELINE_FROM = _rule(
    "ndcf.new_timeline_from", "NDCF distribution", "Date from which the record-date / payment timeline below replaces the 15-day rule",
    date(2024, 11, 27), "date (applied to the declaration date)", "REIT",
    "REIT Regulations, Reg. 18(16)(c) as amended by the Third Amendment Regulations 2024; effective date per SEBI circular "
    "SEBI/HO/DDHS/DDHS-PoD-2/P/CIR/2024/158 of 13 Nov 2024", PARTIAL,
    "Distributions declared on or after this date follow the working-day timeline; earlier ones follow the 15-day rule. "
    "A distribution declared just before the date but paid after it is treated under the 15-day rule.",
)
NDCF_DISTRIBUTION_MAX_DAYS = _rule(
    "ndcf.distribution_days", "NDCF distribution", "Distributions declared before the date above: paid within this many calendar days of declaration",
    15, "days", "REIT", "REIT Regulations, Reg. 18(16)(c) (text as at 23 Oct 2023)", VERIFIED,
    "Late payment carries interest at 15% p.a. (Reg. 18(16)(e)); the interest is not calculated.",
)
NDCF_DISTRIBUTION_MIN_PER_YEAR = _rule(
    "ndcf.distributions_per_year", "NDCF distribution", "Distributions declared at least once every six months in each financial year",
    2, "per financial year", "REIT", "REIT Regulations, Reg. 18(16)(c)", VERIFIED, "Not checked.", enforced=False,
)
NDCF_CF_GAP_MAX_PCT = _rule(
    "ndcf.cf_gap_max", "NDCF distribution", "(CFO + CFI + CFF + PAT) may differ from computed NDCF by at most",
    10.0, "% of computed NDCF", "REIT", "Dashboard heuristic; not in the regulations", HOUSE_RULE,
)
NDCF_RECORD_MAX_WORKING_DAYS = _rule(
    "ndcf.record_working_days", "NDCF distribution", "From the date above: record date within this many working days of declaration",
    2, "working days", "REIT",
    "REIT Regulations, Reg. 18(16)(c) as amended by the Third Amendment Regulations 2024 (figure from published summaries; "
    "amended text not read); see the SEBI circular of 13 Nov 2024", PARTIAL,
    "Working days are counted Monday to Friday, after the declaration date up to and including the record date; market holidays "
    "are not excluded. If the regulation counts differently (for example two clear days between the two dates) a result over "
    "the limit by one day may be on time.",
)
NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS = _rule(
    "ndcf.record_to_distribution_working_days", "NDCF distribution", "From the date above: paid within this many working days of the record date",
    5, "working days", "REIT",
    "REIT Regulations, Reg. 18(16)(c) as amended by the Third Amendment Regulations 2024 (figure from published summaries; "
    "amended text not read); see the SEBI circular of 13 Nov 2024", PARTIAL,
    "Working days are counted Monday to Friday; market holidays are not excluded.",
)

# ---------------------------------------------------------------------------- valuation
VALUER_MAX_TENURE_YEARS = _rule(
    "valuation.tenure_years", "Valuation", "A valuer may not value the same property for more than this many years consecutively",
    4, "years", "REIT", "REIT Regulations, Reg. 21(9)", VERIFIED,
    "Reappointment only after at least 2 years (not checked). The app measures the valuer's tenure per REIT.",
)
VALUATION_REPORT_MAX_DAYS = _rule(
    "valuation.report_days", "Valuation", "Valuation report submitted / disclosed within this many days of the report date",
    15, "days", "REIT", "Earlier dashboard rule (SEBI circular; not checked)", UNVERIFIED,
)
VALUATION_BEFORE_FUNDRAISING_DAYS = _rule(
    "valuation.before_fundraising_days", "Valuation", "A valuation report is needed within this many days before a follow-on fundraising",
    180, "days", "REIT", "Earlier dashboard rule (not checked)", UNVERIFIED,
)

# ------------------------------------------------------------------------ related party
RPT_ACQUISITION_LIMIT = _rule(
    "rpt.acquisition_limit", "Related party", "Acquisition price at most this multiple of the average of the two valuations",
    1.10, "multiple", "REIT",
    "REIT Regulations, Reg. 18(8)(b)(ii): a purchase above 110% of the assessed value needs unitholder approval", PARTIAL,
    "The 110% figure is in the regulation; the 'average of two valuations' form for related-party purchases was not checked.",
)

# ------------------------------------------------------------------------- governance
COMMITTEE_MIN_DIRECTORS = _rule(
    "governance.committee_min_directors", "Governance", "Minimum directors on a committee",
    3, "directors", "REIT manager", "Earlier dashboard rule (not checked)", UNVERIFIED,
)
COMMITTEE_INDEPENDENT_SHARE = _rule(
    "governance.independent_share", "Governance", "Audit / NRC committees: at least this share of directors independent",
    (2, 3), "fraction (numerator, denominator)", "REIT manager", "Earlier dashboard rule (not checked)", UNVERIFIED,
)
COMMITTEE_MIN_INDEPENDENT = _rule(
    "governance.committee_min_independent", "Governance", "Stakeholders Relationship / Risk Management committees: minimum independent directors",
    1, "directors", "REIT manager", "Earlier dashboard rule (not checked)", UNVERIFIED,
)
GOVERNANCE_MEETING_RULES = _rule(
    "governance.committee_meetings", "Governance", "Committee meetings: minimum per year, maximum gap in days, minimum independents present",
    {
        "Audit Committee": {"min_meetings": 4, "gap_days": 120, "min_indep_present": 2},
        "Nomination and Remuneration Committee": {"min_meetings": 1, "min_indep_present": 1},
        "Stakeholders Relationship Committee": {"min_meetings": 1, "min_indep_present": 1},
        "Risk Management Committee": {"min_meetings": 2, "gap_days": 210, "min_indep_present": 1},
    },
    "per committee", "REIT manager", "Earlier dashboard rules (not checked)", UNVERIFIED,
)
BOARD_MIN_MEETINGS = _rule(
    "governance.board_meetings", "Governance", "Board meetings per financial year",
    4, "meetings", "REIT manager", "Earlier dashboard rule (not checked)", UNVERIFIED,
)
BOARD_MAX_GAP_DAYS = _rule(
    "governance.board_gap_days", "Governance", "Maximum gap between board meetings",
    120, "days", "REIT manager", "Earlier dashboard rule (not checked)", UNVERIFIED,
)
BOARD_QUORUM_MIN = _rule(
    "governance.board_quorum_min", "Governance", "Board quorum: at least this many directors, or one third of the board if more",
    3, "directors", "REIT manager", "Earlier dashboard rule (not checked)", UNVERIFIED,
)
BOARD_MIN_INDEPENDENT_PRESENT = _rule(
    "governance.board_independent_present", "Governance", "Independent directors present at each board meeting",
    1, "directors", "REIT manager", "Earlier dashboard rule (not checked)", UNVERIFIED,
)
INDEPENDENT_DIRECTORS_MIN_MEETINGS = _rule(
    "governance.id_meetings", "Governance", "Meetings of independent directors per financial year",
    1, "meetings", "REIT manager", "Earlier dashboard rule (not checked)", UNVERIFIED,
)


def rules_table():
    """Rows for display, grouped by area (used by the Rules reference page and by tests)."""
    return [r for r in RULES.values()]


def display_value(rule: Rule) -> str:
    v = rule.value
    if rule.key == "governance.committee_meetings":
        return "; ".join(f"{k}: {d['min_meetings']}/yr" + (f", gap {d['gap_days']}d" if 'gap_days' in d else "") for k, d in v.items())
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        if rule.unit.startswith("fraction (numerator"):
            return f"{v[0]}/{v[1]}"
        return f"{v[0]:g} to {v[1]:g}"
    if isinstance(v, tuple):
        return "5% / 3% / 2% / 1% by year range"
    if isinstance(v, float) and rule.unit.startswith("fraction"):
        return f"{v * 100:g}%"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)
