"""Tests for the shared sheet-parsing helpers in utils/common.py.

Each test pins a bug that used to produce wrong answers or false alerts on the real sheets.
"""
import numpy as np
import pandas as pd
import pytest

from utils.common import _find_col, _num_series, parse_number, resolve_percent_units


# ------------------------------------------------------------------ _find_col
class TestFindCol:
    COLS = ["Name of REIT", "Financial Year", "Total Value of REIT Assets", "Unitholder Approval Weblink"]

    def test_alias_match_ignores_case_and_punctuation(self):
        assert _find_col(self.COLS, aliases=["name of reit"]) == "Name of REIT"

    def test_missing_alias_without_tokens_is_not_found(self):
        # used to return the FIRST column (silently wrong data after a header rename)
        assert _find_col(self.COLS, aliases=["Name of CRA1"]) is None

    def test_no_aliases_no_tokens_is_not_found(self):
        assert _find_col(self.COLS) is None

    def test_tokens_find_the_column(self):
        assert _find_col(self.COLS, must_tokens=["total value", "reit assets"]) == "Total Value of REIT Assets"

    def test_exclude_tokens(self):
        assert _find_col(self.COLS, must_tokens=["unitholder"], exclude_tokens=["weblink"]) is None

    def test_tokens_that_do_not_match(self):
        assert _find_col(self.COLS, must_tokens=["rating"]) is None


# --------------------------------------------------------------- parse_number
@pytest.mark.parametrize(
    "cell, expected",
    [
        ("1,200", 1200.0),
        ("3,16,124", 316124.0),  # Indian digit grouping
        ("(1,200)", -1200.0),  # accounting negative
        ("−4.5", -4.5),  # unicode minus
        ("₹ 25.5", 25.5),
        ("Rs. 25.5", 25.5),
        ("12%", 12.0),  # % sign ignored: number as written
        (" 7 ", 7.0),
        (3, 3.0),
        (2.5, 2.5),
        ("0", 0.0),  # a real zero stays zero
        (0, 0.0),
    ],
)
def test_parse_number_reads_numbers(cell, expected):
    assert parse_number(cell) == expected


@pytest.mark.parametrize(
    "cell",
    ["", "  ", "-", "—", "NA", "N/A", "n.a.", "nil", "None", "#DIV/0!", "#N/A", "abc", "1.2.3", None, float("nan"), float("inf"), True],
)
def test_parse_number_missing_is_nan_never_zero(cell):
    # used to be 0.0, which turned "no valuation" into a failed check
    assert np.isnan(parse_number(cell))


def test_num_series_missing_column_is_nan_not_zero():
    df = pd.DataFrame({"a": ["1", "2"]})
    s = _num_series(df, None)
    assert s.isna().all() and len(s) == 2


def test_num_series_parses_with_placeholders():
    df = pd.DataFrame({"a": ["1,000", "-", "(5)", "x"]})
    s = _num_series(df, "a")
    assert s.iloc[0] == 1000.0 and np.isnan(s.iloc[1]) and s.iloc[2] == -5.0 and np.isnan(s.iloc[3])


# ------------------------------------------------------- resolve_percent_units
def pct(values, reference=None, groups=None):
    fractions, how = resolve_percent_units(
        pd.Series(values, dtype="object"),
        reference=None if reference is None else pd.Series(reference, dtype="float64"),
        groups=None if groups is None else pd.Series(groups),
    )
    return (fractions * 100).round(4).tolist(), how.tolist()


class TestResolvePercentUnits:
    def test_percent_sign_is_percent_points(self):
        vals, how = pct(["26.09%", "-0.24%"])
        assert vals == [26.09, -0.24] and set(how) == {"with % sign"}

    def test_values_above_one_and_a_half_are_percent_points(self):
        # 6.3 / 56.09 / -6.76 can only be percent points
        vals, _ = pct(["6.3", "56.09", "-6.76"])
        assert vals == [6.3, 56.09, -6.76]

    def test_reference_decides_the_ambiguous_cases(self):
        # PowerGrid typed percent points (0.85 = 0.85%), Energy Infrastructure typed fractions (0.4394 = 43.94%)
        vals, how = pct(["0.85", "0.4394"], reference=[0.0090, 0.4394])
        assert vals == [0.85, 43.94]
        assert all("computed" in h for h in how)

    def test_old_parser_bug_085_is_not_85_percent(self):
        vals, _ = pct(["0.85"], reference=[0.006])
        assert vals[0] == 0.85  # was 85.0, which fired a false "> 70% cap" alert

    def test_group_evidence_when_no_reference(self):
        # same entity has a clear percent-points row (6.3), so its 0.85 is percent points too
        vals, how = pct(["0.85", "6.3", "0.4394"], groups=["P", "P", "E"])
        assert vals == [0.85, 6.3, 43.94]
        assert "same entity" in how[0] and "assumed a fraction" in how[2]

    def test_assumes_fraction_only_as_a_last_resort(self):
        vals, how = pct(["0.5"])
        assert vals == [50.0] and "assumed a fraction" in how[0]

    def test_blank_and_placeholders_stay_missing(self):
        vals, how = pct(["", "#DIV/0!", None, "-"])
        assert all(np.isnan(v) for v in vals) and set(how) == {""}

    def test_mixed_column_from_the_real_sheet(self):
        raw = ["26.09%", "0.85", "1.01", "6.3", "-6.76", "0.4394", "56.09"]
        ref = [0.2609, 0.009, 0.010, np.nan, np.nan, 0.4394, 0.5609]
        grp = ["A", "P", "P", "P", "P", "E", "D"]
        vals, _ = pct(raw, reference=ref, groups=grp)
        assert vals == [26.09, 0.85, 1.01, 6.3, -6.76, 43.94, 56.09]
