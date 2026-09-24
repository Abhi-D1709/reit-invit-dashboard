"""Tests for utils/rules.py and for the pages actually following it.

The point of the config is one source of truth: changing a number in utils/rules.py must change
the behaviour of every page that uses it. These tests change the value at runtime and check that.
"""
import pandas as pd
import pytest

from tabs import governance, investment, ndcf, related_party, valuation
from utils import rules


# ---------------------------------------------------------------------- the registry
class TestRegistry:
    def test_every_rule_is_complete(self):
        assert rules.RULES, "no rules registered"
        for r in rules.RULES.values():
            assert r.key and r.area and r.name and r.unit and r.applies_to and r.source, r.key
            assert r.status in {rules.VERIFIED, rules.PARTIAL, rules.HOUSE_RULE, rules.UNVERIFIED}, r.key

    def test_keys_are_unique(self):
        keys = [r.key for r in rules.RULES.values()]
        assert len(keys) == len(set(keys))

    def test_verified_rules_name_the_regulation(self):
        for r in rules.RULES.values():
            if r.status in (rules.VERIFIED, rules.PARTIAL):
                assert "Reg" in r.source or "circular" in r.source.lower(), f"{r.key}: {r.source}"

    def test_rules_not_checked_by_any_page_say_so(self):
        for r in rules.RULES.values():
            if not r.enforced:
                assert r.note, f"{r.key} is not enforced but has no note explaining why"

    def test_every_rule_can_be_displayed(self):
        for r in rules.RULES.values():
            assert isinstance(rules.display_value(r), str) and rules.display_value(r)

    def test_the_verified_regulatory_figures(self):
        # pinned to the SEBI REIT Regulations text reviewed on 2026-09-24 (consolidated to 23 Oct 2023)
        assert rules.SPONSOR_MIN_INITIAL == 0.15 and rules.SPONSOR_MIN_INITIAL_YEARS == 3.0
        assert rules.PUBLIC_MIN == 0.25 and rules.PUBLIC_MIN_DEADLINE_YEARS == 3.0
        assert rules.REIT_NBR_TRIGGER == 0.25 and rules.REIT_NBR_CAP == 0.49
        assert rules.INVEST_COMPLETED_MIN_PCT == 80.0
        assert rules.NDCF_PAYOUT_MIN_PCT == 90.0 and rules.NDCF_DISTRIBUTION_MAX_DAYS == 15
        assert rules.NDCF_NEW_TIMELINE_FROM.isoformat() == "2024-11-27"
        assert rules.NDCF_RECORD_MAX_WORKING_DAYS == 2 and rules.NDCF_DISTRIBUTION_AFTER_RECORD_MAX_WORKING_DAYS == 5
        assert rules.VALUER_MAX_TENURE_YEARS == 4


# ------------------------------------------------ the pages follow the config at runtime
class TestPagesFollowTheConfig:
    def test_investment_minimum(self, monkeypatch):
        assert investment.asset_ratio_status(75.0, 100.0).startswith("✖")
        monkeypatch.setattr(rules, "INVEST_COMPLETED_MIN_PCT", 70.0)
        assert investment.asset_ratio_status(75.0, 100.0).startswith("✔")

    def test_investment_alert_band_and_its_label(self, monkeypatch):
        monkeypatch.setattr(rules, "INVEST_ALERT_BAND", (70.0, 72.0))
        out = investment.asset_ratio_status(71.0, 100.0)
        assert out.startswith("✖") and "70-72% Bracket" in out
        assert "Bracket" not in investment.asset_ratio_status(83.0, 100.0)  # the old band no longer applies

    def test_related_party_limit(self, monkeypatch):
        assert related_party.acquisition_status(105.0, 100.0, 100.0) == "Pass"
        monkeypatch.setattr(rules, "RPT_ACQUISITION_LIMIT", 1.0)
        assert related_party.acquisition_status(105.0, 100.0, 100.0) == "Fail"

    def test_ndcf_payout_minimum(self, monkeypatch):
        comp = "Total Amount of NDCF computed as per NDCF Statement"
        decl = "Total Amount of NDCF declared for the period (incl. Surplus)"
        df = pd.DataFrame({comp: [100.0], decl: [60.0]})
        cf = {c: [1.0] for c in [
            "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
            "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
            "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
            "Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"]}
        df = pd.concat([df, pd.DataFrame(cf)], axis=1)
        assert ndcf.compute_trust_checks(df)["Meets payout rule"].iloc[0] == False  # noqa: E712
        monkeypatch.setattr(rules, "NDCF_PAYOUT_MIN_PCT", 50.0)
        assert bool(ndcf.compute_trust_checks(df)["Meets payout rule"].iloc[0]) is True

    def test_valuer_tenure_limit(self, monkeypatch):
        rows = pd.DataFrame([{"Name of REIT": "X", "Financial Year": "2023-24", "Name of Valuer": "A", "IBBI Registration No": "R",
                              "Date of Appointment": "01/04/2022", "Date of Resignation": ""}])
        empty = pd.DataFrame(columns=["reg_no", "name", "status"])
        assert bool(valuation.evaluate_rows(rows, empty, empty)["Tenure within limit"].iloc[0]) is True
        monkeypatch.setattr(rules, "VALUER_MAX_TENURE_YEARS", 1)
        out = valuation.evaluate_rows(rows, empty, empty)
        assert out["Tenure within limit"].iloc[0] == False and "> 1 years" in out["Tenure Status"].iloc[0]  # noqa: E712

    @staticmethod
    def _two_directors():
        return pd.DataFrame({
            "Type of Members of Committee": ["Independent Director", "Non-Independent Director"],
            "Role of Members of Committee": ["Non-Executive", "Executive"],
            "Is this Member the Chairperson for the Committee": ["Yes", "No"],
            "Is this member identified as having accounting or related Financial Management Expertise.": ["Yes", "No"],
        })

    @pytest.mark.parametrize("evaluate", ["evaluate_audit", "evaluate_nrc", "evaluate_src", "evaluate_rmc"])
    def test_governance_committee_size_and_label_in_every_committee(self, monkeypatch, evaluate):
        fn = getattr(governance, evaluate)
        two = self._two_directors()
        base = fn(two)
        row = base[base["Check"].str.startswith("Min ")].iloc[0]
        assert row["Check"] == "Min 3 directors" and row["Result"] == "✖ Fail"
        monkeypatch.setattr(rules, "COMMITTEE_MIN_DIRECTORS", 2)
        changed = fn(two)
        row = changed[changed["Check"].str.startswith("Min ")].iloc[0]
        assert row["Check"] == "Min 2 directors" and row["Result"] == "✔ Pass"

    @pytest.mark.parametrize("evaluate", ["evaluate_audit", "evaluate_nrc"])
    def test_governance_independent_share(self, monkeypatch, evaluate):
        fn = getattr(governance, evaluate)
        two = self._two_directors()  # 1 of 2 directors is independent
        base = fn(two)
        share = base[base["Check"].str.contains("independent")].iloc[0]
        assert share["Check"] == "≥ 2/3 independent" and share["Result"] == "✖ Fail"
        monkeypatch.setattr(rules, "COMMITTEE_INDEPENDENT_SHARE", (1, 2))
        changed = fn(two)
        share = changed[changed["Check"].str.contains("independent")].iloc[0]
        assert share["Check"] == "≥ 1/2 independent" and share["Result"] == "✔ Pass"

    @pytest.mark.parametrize("evaluate", ["evaluate_src", "evaluate_rmc"])
    def test_governance_minimum_independent(self, monkeypatch, evaluate):
        fn = getattr(governance, evaluate)
        two = self._two_directors()  # exactly 1 independent
        base = fn(two)
        row = base[base["Check"].str.contains("independent")].iloc[0]
        assert row["Check"] == "≥ 1 independent" and row["Result"] == "✔ Pass"
        monkeypatch.setattr(rules, "COMMITTEE_MIN_INDEPENDENT", 2)
        changed = fn(two)
        row = changed[changed["Check"].str.contains("independent")].iloc[0]
        assert row["Check"] == "≥ 2 independent" and row["Result"] == "✖ Fail"


# ------------------------------------- Reg. 18(16)(c): distribution within 15 days of declaration
def _timeline(decl, rec, dist):
    """One-row frame for compute_trust_timeline_checks; None means a missing date."""
    ts = lambda v: pd.Timestamp(v) if v else pd.NaT  # noqa: E731
    return pd.DataFrame({
        "Financial Year": ["2023-24"], "Period Ended": ["Mar"],
        "Declaration Date": [ts(decl)], "Record Date": [ts(rec)], "Distribution Date": [ts(dist)],
    })


def _row(decl, rec, dist):
    return ndcf.compute_trust_timeline_checks(_timeline(decl, rec, dist)).iloc[0]


class TestDistributionDeadline:
    # the record date sits between declaration and distribution so only the 15-day rule is being exercised
    def _check(self, decl, dist):
        return _row(decl, decl, dist)["Distribution within limit"]

    def test_exactly_fifteen_days_is_on_time(self):
        assert bool(self._check("2024-05-01", "2024-05-16")) is True

    def test_sixteen_days_is_late(self):
        assert self._check("2024-05-01", "2024-05-17") == False  # noqa: E712

    def test_same_day_is_on_time(self):
        assert bool(self._check("2024-05-01", "2024-05-01")) is True

    @pytest.mark.parametrize("decl, rec, dist", [(None, "2024-05-02", "2024-05-16"), ("2024-05-01", "2024-05-02", None), (None, None, None)])
    def test_missing_dates_are_insufficient_not_late(self, decl, rec, dist):
        row = _row(decl, rec, dist)
        assert pd.isna(row["Distribution within limit"]) and row["Date check"] == "Missing date(s)"

    def test_reports_the_number_of_days(self):
        assert _row("2024-05-01", "2024-05-02", "2024-05-11")["Days Decl→Distr"] == 10

    def test_follows_the_config(self, monkeypatch):
        assert self._check("2024-05-01", "2024-05-21") == False  # 20 days  # noqa: E712
        monkeypatch.setattr(rules, "NDCF_DISTRIBUTION_MAX_DAYS", 30)
        assert bool(self._check("2024-05-01", "2024-05-21")) is True

    def test_empty_frame_still_has_the_new_columns(self):
        out = ndcf.compute_trust_timeline_checks(pd.DataFrame({"Financial Year": []}))
        assert {"Days Decl→Distr", "Distribution within limit", "Date check", "Rule applied", "Working days Decl→Record"} <= set(out.columns)

    def test_a_genuinely_late_distribution_is_still_late(self):
        # 29 days after declaration, and no day/month swap would explain it
        row = _row("2024-05-15", "2024-05-16", "2024-06-13")
        assert row["Distribution within limit"] == False and row["Date check"] == ""  # noqa: E712

    def test_dates_that_are_fine_have_no_date_check_message(self):
        assert _row("2024-05-01", "2024-05-02", "2024-05-11")["Date check"] == ""


class TestWorkingDayTimeline:
    """Declarations on or after 27 Nov 2024: record date within 2 working days, payment within 5 working days of it."""

    def test_working_day_count_skips_weekends(self):
        s, e = pd.Series([pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-11"), pd.Timestamp("2025-01-06")]), \
            pd.Series([pd.Timestamp("2025-01-13"), pd.Timestamp("2025-01-13"), pd.Timestamp("2025-01-06")])
        # Fri -> Mon is 1; Sat -> Mon is 1; same day is 0
        assert ndcf.working_days_between(s, e).tolist() == [1.0, 1.0, 0.0]

    def test_working_day_count_missing_and_negative(self):
        out = ndcf.working_days_between(pd.Series([pd.NaT, pd.Timestamp("2025-01-08")]), pd.Series([pd.Timestamp("2025-01-08"), pd.Timestamp("2025-01-06")]))
        assert pd.isna(out.iloc[0]) and out.iloc[1] == -2

    def test_on_time_across_a_weekend(self):
        # declared Thursday, record Monday (2 working days), paid the following Monday (5 working days)
        row = _row("2025-01-09", "2025-01-13", "2025-01-20")
        assert bool(row["Record on time"]) is True and bool(row["Distribution on time"]) is True
        assert row["Days Decl→Record"] == 4 and row["Working days Decl→Record"] == 2  # 4 calendar days, 2 working
        assert row["Rule applied"] == ndcf.RULE_NEW and row["Date check"] == ""

    def test_one_working_day_over_is_late(self):
        row = _row("2025-01-06", "2025-01-09", "2025-01-10")  # record date 3 working days after declaration
        assert row["Record on time"] == False  # noqa: E712
        assert "market holiday" in row["Date check"]

    def test_far_over_has_no_holiday_note(self):
        row = _row("2025-01-06", "2025-01-20", "2025-01-21")  # 10 working days
        assert row["Record on time"] == False and row["Date check"] == ""  # noqa: E712

    def test_the_fifteen_day_check_does_not_apply_to_new_declarations(self):
        row = _row("2025-01-06", "2025-01-07", "2025-01-09")
        assert pd.isna(row["Distribution within limit"])

    def test_the_working_day_checks_do_not_apply_to_old_declarations(self):
        row = _row("2024-05-01", "2024-05-20", "2024-05-30")  # 29 days, far over both new limits
        assert pd.isna(row["Record on time"]) and pd.isna(row["Distribution on time"])
        assert row["Rule applied"] == ndcf.RULE_OLD and row["Distribution within limit"] == False  # noqa: E712

    def test_the_day_before_the_switch_is_the_old_rule(self):
        assert _row("2024-11-26", "2024-11-26", "2024-12-10")["Rule applied"] == ndcf.RULE_OLD
        assert _row("2024-11-27", "2024-11-27", "2024-12-04")["Rule applied"] == ndcf.RULE_NEW

    def test_missing_declaration_date_has_no_rule(self):
        row = _row(None, "2025-01-07", "2025-01-10")
        assert row["Rule applied"] == "" and pd.isna(row["Record on time"]) and row["Date check"] == "Missing date(s)"

    def test_swap_hint_works_under_the_new_timeline(self):
        # payment typed 06/01 was read as 1 Jun; as 6 Jan it is 1 working day after the record date
        row = _row("2025-01-03", "2025-01-06", "2025-06-01")  # payment 1 Jun as entered; swapped it is 6 Jan
        assert row["Distribution on time"] == False  # noqa: E712
        assert row["Date check"] == "Late as entered (a day/month swap would fix it)"


class TestImpossibleDates:
    """Dates out of order are a data-entry problem, never reported as a late payment."""

    def test_distribution_before_declaration_is_not_a_late_payment(self):
        row = _row("2024-05-20", "2024-05-21", "2024-05-01")
        assert pd.isna(row["Distribution within limit"]) and row["Date check"].startswith("Dates out of order")

    def test_an_unreliable_timeline_gives_no_verdict_on_any_of_the_three_checks(self):
        # record date before declaration, yet the distribution is 200 days after declaration:
        # it must not be reported as a 200-day late payment when the record date says the dates are wrong
        row = _row("2024-05-20", "2024-05-10", "2024-12-06")
        assert pd.isna(row["Record on time"]) and pd.isna(row["Distribution on time"]) and pd.isna(row["Distribution within limit"])
        assert row["Date check"].startswith("Dates out of order")

    def test_out_of_order_and_no_swap_explains_it(self):
        row = _row("2023-04-14", "2023-08-23", "2023-04-10")  # 14 Apr cannot be day/month swapped
        assert row["Date check"] == "Dates out of order: check the sheet"


class TestDayMonthSwapHint:
    """The source sheet mixes dd/mm and mm/dd dates; a row that fails only because of that says so."""

    def test_late_as_entered_but_fine_if_swapped(self):
        # as entered: declared 27 Apr, record 5 Jun, distributed 5 Nov (192 days).
        # Swapped: record 6 May, distribution 11 May (14 days after declaration).
        row = _row("2023-04-27", "2023-06-05", "2023-11-05")
        assert row["Distribution within limit"] == False  # noqa: E712
        assert row["Date check"] == "Late as entered (a day/month swap would fix it)"

    def test_out_of_order_and_fixed_by_a_swap(self):
        # distribution (11 Sep) is before declaration (26 Oct); swapping the distribution date to 9 Nov fixes it
        row = _row("2023-10-26", "2023-10-27", "2023-09-11")
        assert row["Date check"] == "Dates out of order: check the sheet (a day/month swap would fix it)"

    def test_swapping_helpers(self):
        assert ndcf._swap_day_month(pd.Timestamp("2023-05-06")) == pd.Timestamp("2023-06-05")
        assert ndcf._swap_day_month(pd.Timestamp("2023-05-27")) is None
        assert ndcf._swap_day_month(pd.NaT) is None


# ------------------------------------------------------------------ the reference page
def test_rules_reference_lists_every_rule():
    from tabs import rules_reference

    df = rules_reference.rules_frame()
    assert len(df) == len(rules.RULES)
    assert set(df["Status"]) <= set(rules_reference.STATUS_ICON.values())
    assert not df["Source"].str.strip().eq("").any()


# ----------------------------------------------------------- governance meetings follow the config
def _board_meetings(dates, present=5, independent=1):
    return pd.DataFrame({
        "Date of Board Meeting": dates,
        "Total No. of Directors Present in the Meeting": [str(present)] * len(dates),
        "Total No. of Independent directors in the meeting": [str(independent)] * len(dates),
    })


class TestBoardAndCommitteeMeetingsFollowTheConfig:
    NO_COMPOSITION = pd.DataFrame(columns=["Type of Committee", "Type of Members of Committee"])

    def test_board_minimum_meetings(self, monkeypatch):
        board = _board_meetings(["01/04/2023", "01/07/2023", "01/10/2023"])  # 3 meetings, gaps well under 120 days
        _, _, ok = governance.evaluate_board_meetings(self.NO_COMPOSITION, board)
        assert ok is False  # the configured minimum is 4
        monkeypatch.setattr(rules, "BOARD_MIN_MEETINGS", 3)
        _, _, ok = governance.evaluate_board_meetings(self.NO_COMPOSITION, board)
        assert ok is True

    def test_board_maximum_gap_and_its_label(self, monkeypatch):
        board = _board_meetings(["01/01/2023", "11/05/2023", "20/09/2023", "30/01/2024"])  # gaps of about 130 days
        summary, _, ok = governance.evaluate_board_meetings(self.NO_COMPOSITION, board)
        assert ok is False
        monkeypatch.setattr(rules, "BOARD_MAX_GAP_DAYS", 200)
        summary, _, ok = governance.evaluate_board_meetings(self.NO_COMPOSITION, board)
        assert ok is True and (summary["Expected"] == 200).any()

    def test_board_quorum_minimum(self, monkeypatch):
        board = _board_meetings(["01/04/2023", "01/07/2023", "01/10/2023", "01/01/2024"], present=3)
        # observed board size 3 -> quorum max(3, 1) = 3: three present is enough
        assert governance.evaluate_board_meetings(self.NO_COMPOSITION, board)[2] is True
        monkeypatch.setattr(rules, "BOARD_QUORUM_MIN", 4)
        assert governance.evaluate_board_meetings(self.NO_COMPOSITION, board)[2] is False

    def test_board_independent_director_present(self, monkeypatch):
        board = _board_meetings(["01/04/2023", "01/07/2023", "01/10/2023", "01/01/2024"], independent=1)
        assert governance.evaluate_board_meetings(self.NO_COMPOSITION, board)[2] is True
        monkeypatch.setattr(rules, "BOARD_MIN_INDEPENDENT_PRESENT", 2)
        assert governance.evaluate_board_meetings(self.NO_COMPOSITION, board)[2] is False

    def test_independent_directors_meeting_minimum(self, monkeypatch):
        one = pd.DataFrame({
            "Date of Meeting of Independent Directors": ["15/03/2024"],
            "Total No. of Independent directors in the meeting": ["3"],
        })
        assert governance.evaluate_independent_directors_meeting_sheet4(one)[2] is True
        monkeypatch.setattr(rules, "INDEPENDENT_DIRECTORS_MIN_MEETINGS", 2)
        assert governance.evaluate_independent_directors_meeting_sheet4(one)[2] is False

    def test_committee_meeting_rules_table(self, monkeypatch):
        comp = pd.DataFrame({"Type of Committee": ["Audit Committee"] * 3,
                             "Type of Members of Committee": ["Independent Director", "Independent Director", "Non-Independent Director"]})
        meetings = pd.DataFrame({
            "Type of Committee": ["Audit Committee"] * 3,
            "Date of Meeting of Committee": ["01/04/2023", "01/07/2023", "01/10/2023"],
            "Total No. of Members Present in the Meeting": ["3", "3", "3"],
            "Total No. of Independent directors in the meeting": ["2", "2", "2"],
        })
        assert governance.evaluate_meetings_for_committee(comp, meetings, "Audit Committee")[2] is False  # needs 4 meetings
        changed = dict(rules.GOVERNANCE_MEETING_RULES)
        changed["Audit Committee"] = {"min_meetings": 3, "gap_days": 120, "min_indep_present": 2}
        monkeypatch.setattr(rules, "GOVERNANCE_MEETING_RULES", changed)
        assert governance.evaluate_meetings_for_committee(comp, meetings, "Audit Committee")[2] is True
