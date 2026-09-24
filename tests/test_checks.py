"""Tests for the compliance checks: missing data must give "insufficient data", never a pass or a fail.

These pin the false alerts that were live on the dashboard (see REVIEW.md).
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tabs import borrowings, investment, ndcf, related_party, sponsor_holding, valuation

nan = np.nan


# ------------------------------------------------------- related party (110% rule)
class TestAcquisitionStatus:
    def test_pass_at_the_limit(self):
        assert related_party.acquisition_status(110.0, 100.0, 100.0) == "Pass"

    def test_fail_above_the_limit(self):
        assert related_party.acquisition_status(110.1, 100.0, 100.0) == "Fail"

    @pytest.mark.parametrize("txn, v1, v2", [(150.4, nan, nan), (150.4, 100.0, nan), (nan, 100.0, 100.0)])
    def test_missing_figures_are_insufficient_not_fail(self, txn, v1, v2):
        # Brookfield 2023-24: valuations were "-" -> read as 0 -> limit 0 -> false "Fail"
        assert related_party.acquisition_status(txn, v1, v2) == "Insufficient data"

    def test_a_real_zero_valuation_is_still_a_number(self):
        assert related_party.acquisition_status(5.0, 0.0, 0.0) == "Fail"


class TestRptTotal:
    def test_sums_readable_amounts_and_counts_the_rest(self):
        total, missing = related_party.rpt_total(pd.Series(["1,000", "-", "500", "n/a"]))
        assert total == 1500.0 and missing == 2

    def test_nothing_readable_is_none_not_zero(self):
        total, missing = related_party.rpt_total(pd.Series(["-", "", None]))
        assert total is None and missing == 3


# ---------------------------------------------------------- investment (80% rule)
class TestAssetRatioStatus:
    def test_missing_completed_assets_is_not_an_alert(self):
        assert investment.asset_ratio_status(nan, 1000.0) == investment.NO_DATA

    def test_missing_or_zero_total_is_no_data(self):
        assert investment.asset_ratio_status(800.0, nan) == investment.NO_DATA
        assert investment.asset_ratio_status(800.0, 0.0) == investment.NO_DATA

    def test_thresholds_as_configured(self):
        below = investment.asset_ratio_status(79.9, 100.0)
        assert below.startswith("🔴") and "< 80%" in below
        assert investment.asset_ratio_status(80.0, 100.0).startswith("🟢")
        assert "81-85%" in investment.asset_ratio_status(83.0, 100.0)
        assert investment.asset_ratio_status(90.0, 100.0).startswith("🟢")

    def test_a_reported_zero_is_a_real_zero(self):
        assert "< 80%" in investment.asset_ratio_status(0.0, 100.0)


# --------------------------------------------------------------------- NDCF
COMP = "Total Amount of NDCF computed as per NDCF Statement"
DECL = "Total Amount of NDCF declared for the period (incl. Surplus)"
CFO = "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
CFI = "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
CFF = "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)"
PAT = "Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)"


def _trust_frame(**over):
    base = {COMP: [100.0], DECL: [95.0], CFO: [50.0], CFI: [-10.0], CFF: [-20.0], PAT: [80.0]}
    base.update({k: [v] for k, v in over.items()})
    return pd.DataFrame(base)


class TestNdcfTrust:
    def test_complete_row(self):
        out = ndcf.compute_trust_checks(_trust_frame())
        assert bool(out["Meets payout rule"].iloc[0]) is True
        assert out["CF Sum"].iloc[0] == 100.0 and bool(out["Within gap limit"].iloc[0]) is True

    def test_declared_below_90_percent_fails(self):
        out = ndcf.compute_trust_checks(_trust_frame(**{DECL: 80.0}))
        assert bool(out["Meets payout rule"].iloc[0]) is False

    def test_missing_declared_amount_is_insufficient_not_a_failure(self):
        out = ndcf.compute_trust_checks(_trust_frame(**{DECL: nan}))
        assert pd.isna(out["Meets payout rule"].iloc[0])

    def test_missing_cash_flow_component_gives_no_made_up_total(self):
        out = ndcf.compute_trust_checks(_trust_frame(**{CFI: nan}))  # used to count as 0
        assert pd.isna(out["CF Sum"].iloc[0]) and pd.isna(out["Within gap limit"].iloc[0])

    def test_non_positive_computed_ndcf_is_insufficient(self):
        out = ndcf.compute_trust_checks(_trust_frame(**{COMP: 0.0}))
        assert pd.isna(out["Meets payout rule"].iloc[0])


class TestNdcfTimeline:
    @staticmethod
    def _frame(decl, rec, dist):
        return pd.DataFrame(
            {
                "Financial Year": ["2023-24"],
                "Period Ended": ["Mar"],
                "Declaration Date": [pd.Timestamp(decl) if decl else pd.NaT],
                "Record Date": [pd.Timestamp(rec) if rec else pd.NaT],
                "Distribution Date": [pd.Timestamp(dist) if dist else pd.NaT],
            }
        )

    # declarations on or after 27 Nov 2024 follow the 2 + 5 working-day timeline
    def test_on_time(self):
        out = ndcf.compute_trust_timeline_checks(self._frame("2025-01-06", "2025-01-07", "2025-01-10"))
        assert bool(out["Record on time"].iloc[0]) and bool(out["Distribution on time"].iloc[0])

    def test_late(self):
        out = ndcf.compute_trust_timeline_checks(self._frame("2025-01-06", "2025-01-15", "2025-02-05"))
        assert out["Record on time"].iloc[0] == False and out["Distribution on time"].iloc[0] == False  # noqa: E712

    def test_missing_dates_are_insufficient_not_late(self):
        out = ndcf.compute_trust_timeline_checks(self._frame("2025-01-06", None, None))
        assert pd.isna(out["Record on time"].iloc[0]) and pd.isna(out["Distribution on time"].iloc[0])


class TestNdcfSpv:
    SPV = [
        "SPV Cash Flow From operating Activities as per Audited/Reviewed",
        "SPV Cash Flow From Investing Activities as per Audited/Reviewed",
        "SPV Cash Flow From Financing Activities as per Audited/Reviewed",
        "SPV Profit after tax as per Audited/Reviewed",
    ]
    HCO = [
        "HoldCo Cash Flow From operating Activities as per Audited/Reviewed",
        "HoldCo Cash Flow From Investing Activities as per Audited/Reviewed",
        "Holdco Cash Flow From Financing Activities as per Audited/Reviewed",
        "Holdco Profit after tax as per Audited/Reviewed",
    ]

    def _frame(self, holdco="HoldCo A", spv=(10, 20, 30, 40), hco=(1, 2, 3, 4)):
        d = {"Name of Holdco (Leave Blank if N/A)": [holdco], COMP: [100.0], DECL: [95.0]}
        d.update({name: [v] for name, v in zip(self.SPV, spv)})
        d.update({name: [v] for name, v in zip(self.HCO, hco)})
        return pd.DataFrame(d)

    def test_with_holdco_adds_both(self):
        assert ndcf.compute_spv_checks(self._frame())["SPV+HoldCo CF Sum"].iloc[0] == 110.0

    def test_no_holdco_blank_holdco_figures_count_as_nothing(self):
        out = ndcf.compute_spv_checks(self._frame(holdco="nan", hco=(nan, nan, nan, nan)))
        assert out["SPV+HoldCo CF Sum"].iloc[0] == 100.0

    def test_missing_spv_figure_gives_no_total(self):
        out = ndcf.compute_spv_checks(self._frame(spv=(10, nan, 30, 40)))
        assert pd.isna(out["SPV+HoldCo CF Sum"].iloc[0]) and pd.isna(out["Within Computed Bound (SPV)"].iloc[0])

    def test_holdco_named_but_figures_missing_gives_no_total(self):
        out = ndcf.compute_spv_checks(self._frame(hco=(1, nan, 3, 4)))
        assert pd.isna(out["SPV+HoldCo CF Sum"].iloc[0])


# ---------------------------------------------------------------- valuation
REGISTRY_COLS = ["reg_no", "name", "status"]


def _valuer_rows(appointment, resignation="", fy="2023-24", name="Asha Rao", reg="IBBI/RV/01/2019/1"):
    return pd.DataFrame(
        [
            {
                "Name of REIT": "X", "Financial Year": fy, "Name of Valuer": name, "IBBI Registration No": reg,
                "Date of Appointment": appointment, "Date of Resignation": resignation,
            }
        ]
    )


class TestValuationTenure:
    IND = pd.DataFrame([{"reg_no": "IBBI/RV/01/2019/1", "name": "Mr. Asha Rao", "status": "Registered"}])
    ENT = pd.DataFrame(columns=REGISTRY_COLS)

    def test_within_four_years(self):
        out = valuation.evaluate_rows(_valuer_rows("01/04/2022"), self.IND, self.ENT)
        assert bool(out["Tenure within limit"].iloc[0]) and out["Tenure Status"].iloc[0].startswith("✅")

    def test_over_four_years(self):
        out = valuation.evaluate_rows(_valuer_rows("01/04/2018"), self.IND, self.ENT)
        assert out["Tenure within limit"].iloc[0] == False and "> 4 years" in out["Tenure Status"].iloc[0]  # noqa: E712

    def test_missing_appointment_date_is_insufficient_not_over_four_years(self):
        out = valuation.evaluate_rows(_valuer_rows(""), self.IND, self.ENT)
        assert pd.isna(out["Tenure within limit"].iloc[0]) and "Insufficient" in out["Tenure Status"].iloc[0]
        assert out[~out["Tenure within limit"].fillna(True)].empty  # not picked up as a breach by the page

    def test_unreadable_financial_year_is_insufficient(self):
        out = valuation.evaluate_rows(_valuer_rows("01/04/2022", fy="FY23"), self.IND, self.ENT)
        assert pd.isna(out["Tenure within limit"].iloc[0])


class TestIbbiRegistryMatching:
    IND = pd.DataFrame(
        [
            {"reg_no": "IBBI/RV/01/2019/1", "name": "Mr. Asha Rao", "status": "Registered"},
            {"reg_no": "IBBI/RV/04/2020/12722", "name": "Sagar Goge", "status": "Registration Cancelled w.e.f. 09 Feb, 2021"},
            {"reg_no": "IBBI/RV/09/2021/77", "name": "Dual Name", "status": "Registration Cancelled w.e.f. 01 Jan, 2020"},
            {"reg_no": "IBBI/RV/09/2022/78", "name": "Dual Name", "status": "Registered"},
        ]
    )
    ENT = pd.DataFrame(columns=REGISTRY_COLS)

    def _run(self, name, reg, ind=None):
        return valuation.evaluate_rows(_valuer_rows("01/04/2022", name=name, reg=reg), self.IND if ind is None else ind, self.ENT)

    def test_found_by_reg_no_case_insensitive(self):
        out = self._run("Anyone", "ibbi/rv/01/2019/1")
        assert bool(out["IBBI Registered?"].iloc[0]) and "Found" in out["IBBI Status"].iloc[0]

    def test_cancelled_is_not_registered_and_says_why(self):
        out = self._run("Sagar Goge", "IBBI/RV/04/2020/12722")
        assert out["IBBI Registered?"].iloc[0] == False and "Cancelled" in out["IBBI Status"].iloc[0]  # noqa: E712

    def test_unknown_is_not_found(self):
        out = self._run("Nobody", "IBBI/RV/00/0000/0")
        assert out["IBBI Registered?"].iloc[0] == False and "Not found" in out["IBBI Status"].iloc[0]  # noqa: E712

    def test_an_active_registration_beats_a_cancelled_one_with_the_same_name(self):
        assert bool(self._run("Dual Name", "")["IBBI Registered?"].iloc[0])

    def test_unavailable_registry_flags_nothing(self):
        out = self._run("Asha Rao", "IBBI/RV/01/2019/1", ind=pd.DataFrame(columns=REGISTRY_COLS))
        assert pd.isna(out["IBBI Registered?"].iloc[0]) and "unavailable" in out["IBBI Status"].iloc[0]
        assert out[~out["IBBI Registered?"].fillna(True)].empty


# ---------------------------------------------------------------- borrowings
BORROWING_COLS = [
    "Name of InvIT", "Financial Year", "Quarter Ended", "Borrowings", "Deferred Payments",
    "Cash and Cash Equivalents", "Value of REIT Assets", "Net Borrowings Ratio (NBR)",
]


def _borrowings_sheet(rows, drop=()):
    return pd.DataFrame(rows, columns=BORROWING_COLS).drop(columns=list(drop))


class TestBorrowingsNbr:
    def test_percent_points_typed_by_one_entity_do_not_become_85_percent(self):
        # PowerGrid-style: 0.85 means 0.85%. The components give ~1%, so the fraction reading (85%) is absurd.
        df = borrowings._process_borrowings_df(_borrowings_sheet([["PowerGrid InvIT", "2022-23", "June", 100.0, 0.0, 90.0, 1000.0, "0.85"]]))
        assert df["NBR_ratio"].iloc[0] == pytest.approx(0.0085)

    def test_fraction_typed_by_another_entity_still_works(self):
        df = borrowings._process_borrowings_df(_borrowings_sheet([["Energy InvIT", "2024-25", "Mar", 500.0, 0.0, 60.0, 1000.0, "0.44"]]))
        assert df["NBR_ratio"].iloc[0] == pytest.approx(0.44)

    def test_negative_percent_points_are_not_minus_676_percent(self):
        df = borrowings._process_borrowings_df(_borrowings_sheet([["PowerGrid InvIT", "2021-22", "Sept", nan, nan, nan, nan, "-6.76"]]))
        assert df["NBR_ratio"].iloc[0] == pytest.approx(-0.0676)

    def test_missing_cash_is_not_treated_as_zero(self):
        # blank sheet NBR and blank cash: cannot be computed (used to compute a ratio with cash = 0)
        df = borrowings._process_borrowings_df(_borrowings_sheet([["X InvIT", "2023-24", "Mar", 500.0, nan, nan, 1000.0, ""]]))
        assert pd.isna(df["NBR_ratio"].iloc[0])

    def test_blank_deferred_payments_means_none(self):
        df = borrowings._process_borrowings_df(_borrowings_sheet([["X InvIT", "2023-24", "Mar", 500.0, nan, 100.0, 1000.0, ""]]))
        assert df["NBR_ratio"].iloc[0] == pytest.approx(0.40)

    def test_missing_borrowings_column_is_not_a_zero_ratio(self):
        df = borrowings._process_borrowings_df(
            _borrowings_sheet([["X InvIT", "2023-24", "Mar", nan, nan, 100.0, 1000.0, ""]], drop=["Borrowings"])
        )
        assert pd.isna(df["NBR_ratio"].iloc[0])

    def test_percent_sign_values_are_unchanged(self):
        df = borrowings._process_borrowings_df(_borrowings_sheet([["X InvIT", "2023-24", "Mar", 100.0, 0.0, 10.0, 1000.0, "26.09%"]]))
        assert df["NBR_ratio"].iloc[0] == pytest.approx(0.2609)


# ------------------------------------------------------------ sponsor holding
class TestSponsorHelpers:
    def test_fy_end(self):
        assert sponsor_holding._fy_end_date("2023-24") == dt.date(2024, 3, 31)
        assert sponsor_holding._fy_end_date("2020") == dt.date(2020, 3, 31)

    @pytest.mark.parametrize("bad", ["", "FY23", "abc", None])
    def test_unreadable_fy_is_none_not_today(self, bad):
        assert sponsor_holding._fy_end_date(bad) is None

    def test_sorting_survives_unreadable_years(self):
        assert sponsor_holding._sort_fy(["2024-25", "abc", "2022-23"]) == ["2022-23", "2024-25", "abc"]
