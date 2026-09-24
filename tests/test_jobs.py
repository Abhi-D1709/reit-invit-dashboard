"""Tests for the data jobs' pure logic (no network)."""
import pandas as pd

from jobs import ingest_uhp


class TestSubmissionOrder:
    def test_later_filing_time_wins(self):
        a = ingest_uhp._submission_order("2022-01-20T10:00:00", "165469")
        b = ingest_uhp._submission_order("2022-01-21T10:00:00", "100")
        assert b > a

    def test_same_time_higher_id_wins(self):
        # NSE listed two BIRET 31-Dec-2021 records with identical broadcast time (ids 165413 and 165469);
        # the winner used to depend on feed order
        lo = ingest_uhp._submission_order("2022-11-17T10:40:01", "165413")
        hi = ingest_uhp._submission_order("2022-11-17T10:40:01", "165469")
        assert hi > lo

    def test_bse_ids_are_ordered_by_their_quarter_code(self):
        assert ingest_uhp._submission_order("2026-07-15T15:50:36", "BSE543225-130") > ingest_uhp._submission_order("2026-07-15T15:50:36", "BSE543225-129")

    def test_missing_id_sorts_lowest(self):
        assert ingest_uhp._submission_order("2022-11-17T10:40:01", "") < ingest_uhp._submission_order("2022-11-17T10:40:01", "1")

    def test_order_is_independent_of_input_order(self):
        rows = [("2022-11-17T10:40:01", "165469"), ("2022-11-17T10:40:01", "165413")]
        assert max(rows, key=lambda r: ingest_uhp._submission_order(*r))[1] == "165469"
        assert max(reversed(rows), key=lambda r: ingest_uhp._submission_order(*r))[1] == "165469"


def test_nse_records_without_xbrl_keep_their_percentages(monkeypatch):
    feed = [
        {"ndsSymbol": "X", "secLname": "X Trust", "secSname": "X", "asOnDate": "31-DEC-2021", "submissionDate": "20-JAN-2022",
         "broadCastDate": "17-NOV-2022 10:40:01", "sponsorGroupPer": "51.5", "publicHoldingPer": "48.5",
         "xbrlFilePath": "https://nsearchives.nseindia.com/corporate/xbrl/null", "ndsID": "1"},
    ]

    class Resp:
        def json(self):
            return feed

    monkeypatch.setattr(ingest_uhp, "_get", lambda kind, url, retries=4: Resp())
    import datetime as dt

    recs = ingest_uhp.nse_records("invits", dt.date(2017, 1, 1), dt.date(2026, 1, 1))
    assert len(recs) == 3  # three feed variants are combined; duplicates are removed later, at merge time
    rec = recs[0]
    assert rec["xbrlFile"] == "" and rec["xbrlFilePath"] == ""
    assert rec["sponsorGroupPer"] == 51.5 and rec["publicHoldingPer"] == 48.5
    assert isinstance(pd.Timestamp(rec["filedAt"]), pd.Timestamp)
