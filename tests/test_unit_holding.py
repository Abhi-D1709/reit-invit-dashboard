"""Tests for the Unit Holding Pattern page's data plumbing.

Pins the production crash where, after a deploy, the page code (new) was fed a cached filings
table (old shape) by Streamlit's cache and failed with KeyError: 'xbrlFile'.
"""
import pandas as pd

from tabs import unit_holding


def _old_shape():
    """What the previous version of the code produced: no `xbrlFile` column, full URLs only."""
    return pd.DataFrame(
        [
            {"ndsSymbol": "A", "secLname": "A Trust", "asOnDate": "30-JUN-2026", "publicHoldingPer": "40", "sponsorGroupPer": "60",
             "xbrlFilePath": "https://nsearchives.nseindia.com/corporate/xbrl/UHP_1_2_WEB.xml", "source": "NSE"},
            {"ndsSymbol": "A", "secLname": "A Trust", "asOnDate": "31-MAR-2021", "publicHoldingPer": "35", "sponsorGroupPer": "65",
             "xbrlFilePath": "https://nsearchives.nseindia.com/corporate/xbrl/null", "source": "NSE"},
        ]
    )


class TestEnsureSchema:
    def test_old_table_gets_xbrl_file_from_the_url(self):
        out = unit_holding._ensure_schema(_old_shape())
        assert out["xbrlFile"].tolist() == ["UHP_1_2_WEB.xml", ""]  # the "null" path means: no XBRL

    def test_new_table_is_left_alone(self):
        df = _old_shape().assign(xbrlFile=["x.xml", ""])
        out = unit_holding._ensure_schema(df)
        assert out["xbrlFile"].tolist() == ["x.xml", ""]

    def test_empty_table_is_left_alone(self):
        assert unit_holding._ensure_schema(pd.DataFrame()).empty

    def test_table_without_any_path_column(self):
        out = unit_holding._ensure_schema(pd.DataFrame({"asOnDate": ["30-JUN-2026"]}))
        assert out["xbrlFile"].tolist() == [""]


class TestCacheDoesNotServeAnOldShape:
    def test_new_data_version_recomputes(self, monkeypatch):
        calls = []

        def fake_fetch(index):
            calls.append(index)
            frame = _old_shape() if len(calls) == 1 else _old_shape().assign(xbrlFile=["x.xml", ""])
            return frame.to_dict("records"), []

        monkeypatch.setattr(unit_holding.filing_source, "fetch_master", fake_fetch)
        unit_holding.get_master_df.clear()

        first, _ = unit_holding.get_master_df("invits", "v1", 2)
        again, _ = unit_holding.get_master_df("invits", "v1", 2)  # same key: served from cache
        assert len(calls) == 1 and again.equals(first)

        fresh, _ = unit_holding.get_master_df("invits", "v2", 2)  # data (or schema) changed: not the stale entry
        assert len(calls) == 2 and "xbrlFile" in fresh.columns

        unit_holding.get_master_df.clear()

    def test_the_result_always_has_the_columns_the_page_uses(self, monkeypatch):
        monkeypatch.setattr(unit_holding.filing_source, "fetch_master", lambda index: (_old_shape().to_dict("records"), []))
        unit_holding.get_master_df.clear()
        df, _ = unit_holding.get_master_df("invits", "v1", 2)
        for col in ("xbrlFile", "entityKey", "asOnDateParsed", "publicHoldingPer", "sponsorGroupPer", "source"):
            assert col in df.columns
        unit_holding.get_master_df.clear()
