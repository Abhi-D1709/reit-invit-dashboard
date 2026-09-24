"""Decision logic must read its thresholds from utils/rules.py.

Each test runs a check, changes one value in utils.rules at runtime, and runs it again: if a page
had a number hard-coded, the result would not change and the test fails. (Display text that merely
quotes a threshold is not covered here.)
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tabs import borrowings, governance, investment, ndcf, sponsor_holding, valuation
from utils import rules
from utils.common import ENT_COL

nan = np.nan


# ------------------------------------------------------------------------------ NDCF
COMP = "Total Amount of NDCF computed as per NDCF Statement"
DECL = "Total Amount of NDCF declared for the period (incl. Surplus)"
CF = [
    "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
    "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
    "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
    "Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)",
]


class TestNdcfThresholds:
    def test_cash_flow_gap_limit(self, monkeypatch):
        df = pd.DataFrame({COMP: [100.0], DECL: [95.0], CF[0]: [120.0], CF[1]: [0.0], CF[2]: [0.0], CF[3]: [0.0]})  # gap = 20%
        assert ndcf.compute_trust_checks(df)["Within gap limit"].iloc[0] == False  # noqa: E712
        monkeypatch.setattr(rules, "NDCF_CF_GAP_MAX_PCT", 25.0)
        assert bool(ndcf.compute_trust_checks(df)["Within gap limit"].iloc[0]) is True

    def _timeline(self, decl, rec, dist):
        frame = pd.DataFrame({
            "Financial Year": ["2023-24"], "Period Ended": ["Mar"],
            "Declaration Date": [pd.Timestamp(decl)], "Record Date": [pd.Timestamp(rec)], "Distribution Date": [pd.Timestamp(dist)],
        })
        return ndcf.compute_trust_timeline_checks(frame).iloc[0]

    def test_record_date_limit(self, monkeypatch):
        # declared Monday 6 Jan 2025, record date Friday 10 Jan: 4 working days
        assert self._timeline("2025-01-06", "2025-01-10", "2025-01-13")["Record on time"] == False  # noqa: E712
        monkeypatch.setattr(rules, "NDCF_RECORD_MAX_WORKING_DAYS", 7)
        assert bool(self._timeline("2025-01-06", "2025-01-10", "2025-01-13")["Record on time"]) is True

    def test_distribution_after_record_limit(self, monkeypatch):
        # record date Tuesday 7 Jan, paid Thursday 16 Jan: 7 working days
        assert self._timeline("2025-01-06", "2025-01-07", "2025-01-16")["Distribution on time"] == False  # noqa: E712
        monkeypatch.setattr(rules, "NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS", 10)
        assert bool(self._timeline("2025-01-06", "2025-01-07", "2025-01-16")["Distribution on time"]) is True

    def test_timeline_switch_date(self, monkeypatch):
        # declared 27 Nov 2024 (the switch date): new rule; the 15-day check does not apply
        row = self._timeline("2024-11-27", "2024-11-28", "2024-12-02")
        assert row["Rule applied"] == ndcf.RULE_NEW and pd.isna(row["Distribution within limit"])
        monkeypatch.setattr(rules, "NDCF_NEW_TIMELINE_FROM", dt.date(2025, 6, 1))
        row = self._timeline("2024-11-27", "2024-11-28", "2024-12-02")
        assert row["Rule applied"] == ndcf.RULE_OLD and pd.isna(row["Record on time"]) and bool(row["Distribution within limit"]) is True

    def test_spv_payout_minimum(self, monkeypatch):
        spv = ["SPV Cash Flow From operating Activities as per Audited/Reviewed", "SPV Cash Flow From Investing Activities as per Audited/Reviewed",
               "SPV Cash Flow From Financing Activities as per Audited/Reviewed", "SPV Profit after tax as per Audited/Reviewed"]
        hco = ["HoldCo Cash Flow From operating Activities as per Audited/Reviewed", "HoldCo Cash Flow From Investing Activities as per Audited/Reviewed",
               "Holdco Cash Flow From Financing Activities as per Audited/Reviewed", "Holdco Profit after tax as per Audited/Reviewed"]
        data = {"Name of Holdco (Leave Blank if N/A)": ["nan"], COMP: [100.0], DECL: [60.0]}
        data.update({c: [1.0] for c in spv})
        data.update({c: [nan] for c in hco})
        df = pd.DataFrame(data)
        assert ndcf.compute_spv_checks(df)["Meets payout rule (SPV)"].iloc[0] == False  # noqa: E712
        monkeypatch.setattr(rules, "NDCF_PAYOUT_MIN_PCT", 50.0)
        assert bool(ndcf.compute_spv_checks(df)["Meets payout rule (SPV)"].iloc[0]) is True

    # a row that is 192 days late as entered but would be 14 days after declaration with day/month swapped
    SWAPPED = ("2023-04-27", "2023-06-05", "2023-11-05")

    def test_swap_hint_uses_the_configured_limit_for_the_corrected_timeline(self, monkeypatch):
        assert self._timeline(*self.SWAPPED)["Date check"] == "Late as entered (a day/month swap would fix it)"
        monkeypatch.setattr(rules, "NDCF_DISTRIBUTION_MAX_DAYS", 10)  # the corrected 14 days would still be late
        assert self._timeline(*self.SWAPPED)["Date check"] == ""

    def test_swap_hint_uses_the_configured_limit_for_the_entered_timeline(self, monkeypatch):
        monkeypatch.setattr(rules, "NDCF_DISTRIBUTION_MAX_DAYS", 300)  # 192 days is now within the limit
        assert self._timeline(*self.SWAPPED)["Date check"] == ""


# ------------------------------------------------------------------------- valuation
class TestValuationThresholds:
    def _frames(self):
        report = "Date of valuation report from valuer"
        trustee = "Date of submission of Valuation Report to Trustee"
        df = pd.DataFrame({
            "Name of REIT": ["Alpha REIT"], "Financial Year": ["2023-24"], "Frequency": ["Annual"], "Period Ended": ["Mar"],
            report: ["11/12/2023"], trustee: ["30/12/2023"],  # 19 days after the report
        })
        fund = pd.DataFrame({ENT_COL: ["Alpha REIT"], "Type of Issue": ["Follow-on"], "FundDate": [pd.Timestamp("2024-06-30")]})  # 202 days after
        return df, fund

    def test_report_submission_limit(self, monkeypatch):
        df, fund = self._frames()
        assert "✖" in valuation.check_timelines_and_completeness(df, fund)[0]["Check: Trustee Submission"].iloc[0]
        monkeypatch.setattr(rules, "VALUATION_REPORT_MAX_DAYS", 30)
        assert "✔" in valuation.check_timelines_and_completeness(df, fund)[0]["Check: Trustee Submission"].iloc[0]

    def test_valuation_before_fundraising_window(self, monkeypatch):
        df, fund = self._frames()
        assert valuation.check_timelines_and_completeness(df, fund)[2]["Status"].iloc[0].endswith("Fail")
        monkeypatch.setattr(rules, "VALUATION_BEFORE_FUNDRAISING_DAYS", 250)
        assert valuation.check_timelines_and_completeness(df, fund)[2]["Status"].iloc[0].endswith("Pass")


# -------------------------------------------------------------------------- investment
class TestSpvHoldingLimit:
    def test_limit_and_label(self, monkeypatch):
        assert investment.spv_holding_status({"Shareholder A": 60.0}).startswith("✖")
        assert investment.spv_holding_status({"Shareholder A": 50.0}).startswith("✔")  # at the limit is fine
        monkeypatch.setattr(rules, "SPV_HOLDING_MAX_PCT", 70.0)
        out = investment.spv_holding_status({"Shareholder A": 60.0})
        assert out.startswith("✔") and "70%" in out

    def test_no_figures_is_no_data_not_a_pass(self):
        assert investment.spv_holding_status({}).startswith("?")


# -------------------------------------------------------------------------- borrowings
class TestBorrowingsThresholds:
    def test_invit_cap(self, monkeypatch):
        assert borrowings.nbr_over_cap(0.75, "invit") is True
        monkeypatch.setattr(rules, "INVIT_NBR_CAP", 0.80)
        assert borrowings.nbr_over_cap(0.75, "invit") is False

    def test_reit_cap(self, monkeypatch):
        assert borrowings.nbr_over_cap(0.50, "reit") is True
        assert borrowings.nbr_over_cap(0.49, "reit") is False  # "never exceed": exactly at the cap is fine
        assert borrowings.nbr_over_cap(0.395, "reit") is False
        monkeypatch.setattr(rules, "REIT_NBR_CAP", 0.55)
        assert borrowings.nbr_over_cap(0.50, "reit") is False
        assert borrowings.nbr_cap("reit") == 0.55 and borrowings.nbr_cap("invit") == rules.INVIT_NBR_CAP

    def test_invit_rating_trigger(self, monkeypatch):
        assert borrowings.compliance_sections_required(0.30, "invit") is True
        monkeypatch.setattr(rules, "INVIT_RATING_TRIGGER", 0.35)
        assert borrowings.compliance_sections_required(0.30, "invit") is False

    def test_reit_trigger_is_inclusive(self, monkeypatch):
        assert borrowings.compliance_sections_required(0.25, "reit") is True  # Reg. 20(3): "exceed" is applied inclusively here
        assert borrowings.compliance_sections_required(0.24, "reit") is False
        monkeypatch.setattr(rules, "REIT_NBR_TRIGGER", 0.30)
        assert borrowings.compliance_sections_required(0.27, "reit") is False

    def test_invit_aaa_tier(self, monkeypatch):
        # 55% NBR, a rating exists but is not AAA, no unitholder approval
        assert borrowings.missing_compliance_items(0.55, "invit", True, False, False) == ["AAA Credit Rating", "Unitholder Approval"]
        monkeypatch.setattr(rules, "INVIT_AAA_THRESHOLD", 0.60)  # 55% now only needs a rating, which exists
        assert borrowings.missing_compliance_items(0.55, "invit", True, False, False) == ["Unitholder Approval"]

    def test_invit_rating_tier(self, monkeypatch):
        assert borrowings.missing_compliance_items(0.30, "invit", False, False, True) == ["Credit Rating"]
        monkeypatch.setattr(rules, "INVIT_RATING_TRIGGER", 0.35)
        assert borrowings.missing_compliance_items(0.30, "invit", False, False, False) == []

    def test_reit_always_needs_both_once_the_sections_apply(self):
        assert borrowings.missing_compliance_items(0.30, "reit", False, False, False) == ["Credit Rating", "Unitholder Approval"]


# ---------------------------------------------------------------------- sponsor holding
LISTED = dt.date(2023, 4, 1)
FY_END_2Y = dt.date(2024, 3, 31)  # about 1 year after listing
FY_END_OLD = dt.date(2027, 3, 31)  # about 4 years after listing


class TestSponsorHoldingThresholds:
    def test_initial_sponsor_minimum(self, monkeypatch):
        level, _, within = sponsor_holding.sponsor_public_status(0.10, 0.90, LISTED, FY_END_2Y)
        assert (level, within) == ("error", True)
        monkeypatch.setattr(rules, "SPONSOR_MIN_INITIAL", 0.05)
        assert sponsor_holding.sponsor_public_status(0.10, 0.90, LISTED, FY_END_2Y)[0] == "success"

    def test_initial_period_length(self, monkeypatch):
        assert sponsor_holding.sponsor_public_status(0.10, 0.90, LISTED, FY_END_2Y)[2] is True
        monkeypatch.setattr(rules, "SPONSOR_MIN_INITIAL_YEARS", 0.5)  # the sponsor rule has already ended
        monkeypatch.setattr(rules, "PUBLIC_MIN_DEADLINE_YEARS", 0.5)
        assert sponsor_holding.sponsor_public_status(0.10, 0.90, LISTED, FY_END_2Y)[2] is False

    def test_public_minimum(self, monkeypatch):
        level, message, within = sponsor_holding.sponsor_public_status(0.85, 0.20, LISTED, FY_END_OLD)
        assert (level, within) == ("error", False) and "25%" in message
        monkeypatch.setattr(rules, "PUBLIC_MIN", 0.15)
        assert sponsor_holding.sponsor_public_status(0.85, 0.20, LISTED, FY_END_OLD)[0] == "success"

    def test_public_deadline_is_its_own_setting(self, monkeypatch):
        monkeypatch.setattr(rules, "PUBLIC_MIN_DEADLINE_YEARS", 6.0)  # deadline not reached 4 years after listing
        level, message, within = sponsor_holding.sponsor_public_status(0.85, 0.20, LISTED, FY_END_OLD)
        assert level == "info" and "nothing is checked" in message and within is False

    def test_missing_data_and_unreadable_inputs(self):
        assert sponsor_holding.sponsor_public_status(nan, 0.5, LISTED, FY_END_2Y)[0] == "info"
        assert sponsor_holding.sponsor_public_status(0.5, nan, LISTED, FY_END_OLD)[0] == "info"
        assert sponsor_holding.sponsor_public_status(0.5, 0.5, None, FY_END_2Y)[0] == "warning"
        assert sponsor_holding.sponsor_public_status(0.5, 0.5, pd.NaT, FY_END_2Y)[0] == "warning"  # a blank sheet cell
        assert sponsor_holding.sponsor_public_status(0.5, 0.5, LISTED, None, "FY??")[0] == "warning"


# ------------------------------------------------------------------------- governance
class TestGovernanceThresholds:
    @staticmethod
    def _members(independent, total):
        types = ["Independent Director"] * independent + ["Non-Independent Director"] * (total - independent)
        return pd.DataFrame({
            "Type of Members of Committee": types,
            "Role of Members of Committee": ["Non-Executive"] * total,
            "Is this Member the Chairperson for the Committee": ["Yes"] + ["No"] * (total - 1),
            "Is this member identified as having accounting or related Financial Management Expertise.": ["Yes"] * total,
        })

    @pytest.mark.parametrize("evaluate", ["evaluate_audit", "evaluate_nrc"])
    @pytest.mark.parametrize("share, independent, total, expected", [
        ((3, 4), 2, 4, "✖ Fail"),  # 2 of 4 = 50% < 3/4; a hard-coded numerator of 2 would pass this (8 >= 8)
        ((1, 2), 1, 3, "✖ Fail"),  # 1 of 3 < 1/2; a hard-coded denominator of 3 would pass this (3 >= 3)
        ((1, 2), 2, 3, "✔ Pass"),
    ])
    def test_independent_share_numerator_and_denominator(self, monkeypatch, evaluate, share, independent, total, expected):
        monkeypatch.setattr(rules, "COMMITTEE_INDEPENDENT_SHARE", share)
        out = getattr(governance, evaluate)(self._members(independent, total))
        assert out[out["Check"].str.contains("independent")].iloc[0]["Result"] == expected

    def _committee_meetings(self, independents_present):
        comp = pd.DataFrame({"Type of Committee": ["Audit Committee"] * 3,
                             "Type of Members of Committee": ["Independent Director"] * 3})
        meetings = pd.DataFrame({
            "Type of Committee": ["Audit Committee"] * 4,
            "Date of Meeting of Committee": ["01/04/2023", "01/07/2023", "01/10/2023", "01/01/2024"],
            "Total No. of Members Present in the Meeting": ["3"] * 4,
            "Total No. of Independent directors in the meeting": [str(independents_present)] * 4,
        })
        return comp, meetings

    def test_minimum_independents_present_at_a_committee_meeting(self, monkeypatch):
        comp, meetings = self._committee_meetings(independents_present=2)
        assert governance.evaluate_meetings_for_committee(comp, meetings, "Audit Committee")[2] is True
        changed = {**rules.GOVERNANCE_MEETING_RULES, "Audit Committee": {"min_meetings": 4, "gap_days": 120, "min_indep_present": 3}}
        monkeypatch.setattr(rules, "GOVERNANCE_MEETING_RULES", changed)
        assert governance.evaluate_meetings_for_committee(comp, meetings, "Audit Committee")[2] is False

    def test_maximum_gap_between_committee_meetings(self, monkeypatch):
        comp, meetings = self._committee_meetings(independents_present=2)
        meetings["Date of Meeting of Committee"] = ["01/01/2023", "20/06/2023", "01/12/2023", "01/06/2024"]  # gaps of about 170 days
        assert governance.evaluate_meetings_for_committee(comp, meetings, "Audit Committee")[2] is False
        changed = {**rules.GOVERNANCE_MEETING_RULES, "Audit Committee": {"min_meetings": 4, "gap_days": 200, "min_indep_present": 2}}
        monkeypatch.setattr(rules, "GOVERNANCE_MEETING_RULES", changed)
        assert governance.evaluate_meetings_for_committee(comp, meetings, "Audit Committee")[2] is True
