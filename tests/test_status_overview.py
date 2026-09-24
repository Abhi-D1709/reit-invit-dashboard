"""Shared status vocabulary, period helpers, the overview, the freshness line and the theme."""
import datetime as dt
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabs import borrowings, investment, ndcf, overview, sponsor_holding
from utils import chrome, periods, rules, status
from utils.common import ENT_COL, FY_COL, QTR_COL
from utils.status import CheckResult, Status

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- status
class TestStatus:
    def test_every_status_has_a_glyph_and_a_word(self):
        for s in Status:
            assert status.GLYPH[s] and status.LABEL[s] and status.ICON[s] and status.ALERT[s]
        assert len({status.GLYPH[s] for s in Status}) == len(Status), "glyphs must differ so status never relies on colour"

    def test_tag_and_tagged(self):
        assert status.tag(Status.FAIL) == "✖ Fail"
        assert status.tag(Status.PASS, "12% (no alert)") == "✔ 12% (no alert)"
        assert status.tagged(Status.FAIL, "75% (< 80%)") == "✖ Fail · 75% (< 80%)"

    @pytest.mark.parametrize("value, expected", [
        (True, Status.PASS), (False, Status.FAIL), (np.True_, Status.PASS), (np.False_, Status.FAIL),
        (None, Status.NO_DATA), (np.nan, Status.NO_DATA), (pd.NA, Status.NO_DATA),
    ])
    def test_from_bool_missing_is_no_data_never_a_pass_or_fail(self, value, expected):
        assert status.from_bool(value) is expected

    def test_of_text_reads_back_what_tag_wrote(self):
        for s in Status:
            assert status.of_text(status.tag(s, "something")) is s
            assert status.of_text(status.tagged(s, "something")) is s
        assert status.of_text("plain text") is None

    def test_has_status(self):
        col = pd.Series([status.tagged(Status.PASS, "a"), status.tagged(Status.FAIL, "b")])
        assert status.has_status(col, Status.FAIL) and not status.has_status(col, Status.NO_DATA)

    def test_worst_orders_by_severity(self):
        assert status.worst([Status.PASS, Status.NO_DATA]) is Status.NO_DATA  # missing data is never reported as a pass
        assert status.worst([Status.PASS, Status.REVIEW, Status.NO_DATA]) is Status.REVIEW
        assert status.worst([Status.PASS, Status.FAIL, Status.REVIEW]) is Status.FAIL
        assert status.worst([Status.NA, Status.PASS]) is Status.PASS
        assert status.worst([]) is Status.NA

    def test_badge_shows_glyph_and_word_and_escapes_text(self):
        html = status.badge_html(Status.FAIL, "<b>x</b>")
        assert "✖" in html and "badge-fail" in html and "<b>" not in html and "&lt;b&gt;" in html
        assert "Pass" in status.badge_html(Status.PASS)

    def test_from_alert_covers_every_alert_name(self):
        assert [status.from_alert(a) for a in ("success", "error", "warning", "info")] == [Status.PASS, Status.FAIL, Status.REVIEW, Status.NO_DATA]


# --------------------------------------------------------------------------- periods
class TestPeriods:
    def test_fy_end_date(self):
        assert periods.fy_end_date("2024-25") == dt.date(2025, 3, 31)
        assert periods.fy_end_date("2020") == dt.date(2020, 3, 31)
        assert periods.fy_end_date("abc") is None and periods.fy_end_date(None) is None

    def test_latest_fy_ignores_unreadable_values(self):
        assert periods.latest_fy(["2022-23", "2025-26", "2024-25", "n/a"]) == "2025-26"
        assert periods.latest_fy(["n/a"]) is None and periods.latest_fy([]) is None

    def test_latest_quarter_follows_the_financial_year(self):
        assert periods.latest_quarter(["Jun", "Sept", "Dec"]) == "Dec"
        assert periods.latest_quarter(["Mar", "Jun"]) == "Mar"
        assert periods.latest_quarter(["June"]) == "June"
        assert periods.latest_quarter([]) is None

    def test_sort_fy(self):
        assert periods.sort_fy(["2024-25", "abc", "2022-23"]) == ["2022-23", "2024-25", "abc"]


# -------------------------------------------------------------------------- overview
def _r(st_, key=""):
    return CheckResult("c", st_, "msg", "Area", key)


class TestOverviewLogic:
    def test_a_failure_on_an_unverified_or_partial_rule_is_shown_as_review(self):
        partial = next(k for k, r in rules.RULES.items() if r.status == rules.PARTIAL)
        unverified = next(k for k, r in rules.RULES.items() if r.status == rules.UNVERIFIED)
        for key in (partial, unverified):
            out = overview._by_rule_confidence(_r(Status.FAIL, key))
            assert out.status is Status.REVIEW and "Rules reference" in out.message

    def test_verified_and_house_rules_and_passes_are_left_alone(self):
        verified = next(k for k, r in rules.RULES.items() if r.status == rules.VERIFIED)
        house = next(k for k, r in rules.RULES.items() if r.status == rules.HOUSE_RULE)
        assert overview._by_rule_confidence(_r(Status.FAIL, verified)).status is Status.FAIL
        assert overview._by_rule_confidence(_r(Status.FAIL, house)).status is Status.FAIL
        assert overview._by_rule_confidence(_r(Status.PASS, next(iter(rules.RULES)))).status is Status.PASS
        assert overview._by_rule_confidence(_r(Status.FAIL, "")).status is Status.FAIL  # no rule attached

    def test_an_adapter_that_raises_becomes_no_data_with_the_reason(self):
        def boom(entity):
            raise ValueError("sheet is down")

        out = overview.run_area("X REIT", "Area", boom)
        assert len(out) == 1 and out[0].status is Status.NO_DATA and "sheet is down" in out[0].message

    def test_an_adapter_returning_nothing_is_no_data_not_a_pass(self):
        assert overview.run_area("X", "Area", lambda e: [])[0].status is Status.NO_DATA

    def test_counts_and_area_status(self):
        collected = {"A": {"one": [_r(Status.PASS), _r(Status.FAIL)], "two": [_r(Status.REVIEW)]}, "B": {"one": [_r(Status.NO_DATA)]}}
        n = overview.counts(collected)
        assert (n[Status.PASS], n[Status.FAIL], n[Status.REVIEW], n[Status.NO_DATA]) == (1, 1, 1, 1)
        assert overview.area_status(collected["A"]["one"]) is Status.FAIL

    def test_details_are_sorted_most_severe_first(self):
        by_area = {"a": [_r(Status.PASS), _r(Status.NO_DATA), _r(Status.FAIL), _r(Status.REVIEW)]}
        assert list(overview.details_frame(by_area)["Result"]) == ["✖ Fail", "▲ Review", "? No data", "✔ Pass"]

    def test_details_show_the_regulatory_basis(self):
        key = "borrowings.reit_cap"
        frame = overview.details_frame({"a": [CheckResult("cap", Status.PASS, "ok", "Borrowings", key)]})
        assert frame["Basis"].iloc[0] == rules.RULES[key].source

    def test_matrix_names_every_verdict_in_words_and_escapes_entity_names(self):
        collected = {"A <b>&</b> REIT": {area: [_r(Status.FAIL)] for area, _, _ in overview.AREAS}}
        html = overview.matrix_html(collected)
        assert html.count("Fail") == len(overview.AREAS) and "<b>" not in html and "&lt;b&gt;" in html
        assert 'scope="row"' in html and 'scope="col"' in html and "<caption" in html

    def test_every_overview_area_has_a_page(self):
        from utils import navigation

        groups = navigation.build()
        pages = [p for section in groups.values() for p in section]
        assert len(pages) == len(navigation.PAGE_BY_PATH), "page URLs must be unique"
        assert {path for _, _, path in overview.AREAS} <= set(navigation.PAGE_BY_PATH)


# ------------------------------------------------ the per-area adapters, on synthetic sheets
def _borrowing_frame(nbr, **extra):
    row = {ENT_COL: "X REIT", FY_COL: "2025-26", QTR_COL: "Jun", "NBR_ratio": nbr, **extra}
    return pd.DataFrame([{ENT_COL: "X REIT", FY_COL: "2024-25", QTR_COL: "Mar", "NBR_ratio": 0.60}, row])


class TestBorrowingsSummary:
    def test_over_the_cap_fails_using_the_latest_quarter_only(self, monkeypatch):
        monkeypatch.setattr(borrowings, "load_borrowings_url", lambda url: _borrowing_frame(0.55))
        out = borrowings.summary_results("X REIT")
        cap = next(r for r in out if "cap" in r.check)
        assert cap.status is Status.FAIL and "2025-26 Jun" in cap.message

    def test_below_the_trigger_only_the_cap_check_runs(self, monkeypatch):
        monkeypatch.setattr(borrowings, "load_borrowings_url", lambda url: _borrowing_frame(0.20))
        out = borrowings.summary_results("X REIT")
        assert [r.status for r in out] == [Status.PASS]

    def test_missing_nbr_is_no_data(self, monkeypatch):
        monkeypatch.setattr(borrowings, "load_borrowings_url", lambda url: _borrowing_frame(np.nan))
        assert borrowings.summary_results("X REIT")[0].status is Status.NO_DATA

    def test_unknown_entity_is_no_data(self, monkeypatch):
        monkeypatch.setattr(borrowings, "load_borrowings_url", lambda url: _borrowing_frame(0.2))
        assert borrowings.summary_results("Nobody REIT")[0].status is Status.NO_DATA

    def test_high_nbr_without_rating_or_approval_fails(self, monkeypatch):
        monkeypatch.setattr(borrowings, "load_borrowings_url", lambda url: _borrowing_frame(0.30))
        out = borrowings.summary_results("X REIT")
        needed = next(r for r in out if "rating" in r.check.lower())
        assert needed.status is Status.FAIL and "missing" in needed.message


class TestSponsorSummary:
    def _frame(self, listing):
        return pd.DataFrame([{ENT_COL: "X REIT", FY_COL: "2024-25", "Sponsor+Group %": 0.10, "Public %": 0.90, "__listing_dt__": listing}])

    def test_a_blank_listing_date_is_flagged_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(sponsor_holding, "_load_sponsor_df", lambda url: self._frame(pd.NaT))
        out = sponsor_holding.summary_results("X REIT")
        assert out[0].status is Status.REVIEW and "listing date" in out[0].message

    def test_public_holding_below_the_minimum_fails(self, monkeypatch):
        frame = self._frame(dt.date(2015, 1, 1))
        frame["Public %"] = 0.20
        monkeypatch.setattr(sponsor_holding, "_load_sponsor_df", lambda url: frame)
        assert sponsor_holding.summary_results("X REIT")[0].status is Status.FAIL


class TestInvestmentSummary:
    def _frame(self, completed, total, mutual_funds="0"):
        return pd.DataFrame([{
            "Name of REIT": "X REIT", "Financial Year": "2025-26",
            "Value of completed and rent generating investments": completed, "Total value of REIT assets": total,
            "Investment in mutual funds credit risk": mutual_funds,
        }])

    def test_ratio_below_the_minimum_fails(self, monkeypatch):
        monkeypatch.setattr(investment, "load_investment_data", lambda: self._frame(70, 100))
        assert investment.summary_results("X REIT")[0].status is Status.FAIL

    def test_missing_figures_are_no_data_not_a_failure(self, monkeypatch):
        monkeypatch.setattr(investment, "load_investment_data", lambda: self._frame("-", "-"))
        assert investment.summary_results("X REIT")[0].status is Status.NO_DATA

    def test_mutual_funds_found_need_review(self, monkeypatch):
        monkeypatch.setattr(investment, "load_investment_data", lambda: self._frame(90, 100, mutual_funds="12.5"))
        assert any(r.check == "Mutual fund investments" and r.status is Status.REVIEW for r in investment.summary_results("X REIT"))


class TestNdcfSummary:
    COMP = "Total Amount of NDCF computed as per NDCF Statement"
    DECL = "Total Amount of NDCF declared for the period (incl. Surplus)"
    CF = [
        "Cash Flow From operating Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "Cash Flow From Investing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "Cash Flow From Financing Activities as per Cash Flow Statements (as per Audited Financials or Financials with Limited Review)",
        "Profit after tax as per Statement of Profit and Loss (as per Audited Financials or Financials with Limited Review)",
    ]

    def _frame(self, decl, rec, dist, computed=100.0, declared=95.0):
        ts = lambda v: pd.Timestamp(v) if v else pd.NaT  # noqa: E731
        row = {"Name of REIT": "X REIT", "Financial Year": "2025-26", "Period Ended": "Jun", self.COMP: computed, self.DECL: declared,
               "Declaration Date": ts(decl), "Record Date": ts(rec), "Distribution Date": ts(dist)}
        row.update({c: 25.0 for c in self.CF})
        return pd.DataFrame([row])

    def _run(self, monkeypatch, frame):
        monkeypatch.setattr(ndcf, "_read_trust_df_from_gsheet", lambda url: frame)
        return {r.check.split(" ")[0]: r for r in ndcf.summary_results("X REIT")}

    def test_all_good(self, monkeypatch):
        out = self._run(monkeypatch, self._frame("2025-01-06", "2025-01-08", "2025-01-13"))
        assert {r.status for r in out.values()} == {Status.PASS}

    def test_late_payment_names_the_limit_that_was_missed(self, monkeypatch):
        out = self._run(monkeypatch, self._frame("2025-01-06", "2025-01-08", "2025-01-29"))
        timeline = out["Distribution"]
        assert timeline.status is Status.FAIL and "after the record date (limit 5)" in timeline.message

    def test_missing_dates_are_no_data_and_do_not_crash(self, monkeypatch):
        # regression: comparing <NA> with False raised "boolean value of NA is ambiguous"
        out = self._run(monkeypatch, self._frame("2025-01-06", None, None))
        assert out["Distribution"].status is Status.NO_DATA

    def test_payout_below_ninety_percent_fails(self, monkeypatch):
        out = self._run(monkeypatch, self._frame("2025-01-06", "2025-01-08", "2025-01-13", declared=50.0))
        assert out["Payout"].status is Status.FAIL

    def test_the_cash_flow_gap_is_a_review_not_a_fail_because_it_is_a_house_rule(self, monkeypatch):
        frame = self._frame("2025-01-06", "2025-01-08", "2025-01-13")
        frame[self.CF[0]] = 500.0
        out = self._run(monkeypatch, frame)
        assert out["Cash-flow"].status is Status.REVIEW


# ------------------------------------------------------------------------- freshness
class TestDataAsOf:
    MANIFEST = {
        "trades": {"last_date": "2026-09-23"}, "uhp": {"generated_at": "2026-09-24T09:51:22Z", "errors": []},
        "ibbi": {"generated_at": "2026-09-24T06:44:33Z", "errors": []}, "errors": [],
    }

    def test_reports_each_source_date(self):
        text = chrome.data_as_of_text(self.MANIFEST)
        assert "through 23 Sep 2026" in text and "as of 24 Sep 2026" in text and "Google Sheets" in text
        assert "error" not in text

    def test_job_errors_are_surfaced(self):
        text = chrome.data_as_of_text({**self.MANIFEST, "errors": ["boom"]})
        assert "1 data-job error" in text

    def test_an_empty_manifest_still_gives_a_line(self):
        assert "Google Sheets" in chrome.data_as_of_text({})


# ------------------------------------------------------------------------------ theme
def _luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class TestTheme:
    THEME = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]

    @pytest.mark.parametrize("mode", ["light", "dark"])
    def test_text_and_accent_colours_meet_wcag_aa_on_both_backgrounds(self, mode):
        t = self.THEME[mode]
        for bg in (t["backgroundColor"], t["secondaryBackgroundColor"]):
            for fg in ("textColor", "primaryColor", "linkColor"):
                assert contrast(t[fg], bg) >= 4.5, f"{mode}: {fg} {t[fg]} on {bg} is {contrast(t[fg], bg):.2f}:1"

    def test_the_chart_colours_follow_the_theme(self, monkeypatch):
        from utils import theme

        assert theme.is_dark() is False  # outside a running app
        light = (theme.text_color(), theme.accent(), theme.table_row_styles()["total"])
        monkeypatch.setattr(theme, "is_dark", lambda: True)
        assert (theme.text_color(), theme.accent(), theme.table_row_styles()["total"]) != light

    @pytest.mark.parametrize("dark", [False, True])
    def test_chart_text_is_readable_on_the_matching_background(self, monkeypatch, dark):
        from utils import theme

        monkeypatch.setattr(theme, "is_dark", lambda: dark)
        bg = self.THEME["dark" if dark else "light"]["backgroundColor"]
        assert contrast(theme.text_color(), bg) >= 4.5 and contrast(theme.accent(), bg) >= 3.0
