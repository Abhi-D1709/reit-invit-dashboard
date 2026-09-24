# utils/periods.py
"""Financial-year and quarter helpers (Indian FY: April to March), shared by pages and the scorecard."""
from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

_QUARTER_RANK = {"jun": 0, "june": 0, "sep": 1, "sept": 1, "september": 1, "dec": 2, "december": 2, "mar": 3, "march": 3}


def fy_end_date(fy) -> Optional[date]:
    """'2019-20' -> 2020-03-31, '2024-25' -> 2025-03-31, '2020' -> 2020-03-31.
    None when the text isn't a financial year (it must never default to today's date)."""
    fy = str(fy or "").strip()
    try:
        if "-" in fy:
            _, b = fy.split("-", 1)
            end_year = int("20" + b[-2:]) if len(b) == 2 else int(b)
        else:
            end_year = int(fy)
        return date(end_year, 3, 31)
    except Exception:
        return None


def sort_fy(values: Iterable) -> list:
    """Financial years oldest first; unreadable ones go last."""
    return sorted(values, key=lambda fy: fy_end_date(str(fy)) or date.max)


def latest_fy(values: Iterable) -> Optional[str]:
    """The most recent readable financial year, or None."""
    dated = [(fy_end_date(str(v)), str(v)) for v in values if v is not None]
    dated = [(d, v) for d, v in dated if d is not None]
    return max(dated)[1] if dated else None


def quarter_rank(q) -> int:
    """Position of a quarter label within the financial year (Jun=0 ... Mar=3); 99 if unknown."""
    return _QUARTER_RANK.get(str(q).strip().lower(), 99)


def latest_quarter(values: Iterable) -> Optional[str]:
    """The last quarter of the year among the labels (Mar after Dec after Sept after Jun)."""
    labels = [str(v) for v in values if v is not None]
    known = [v for v in labels if quarter_rank(v) != 99]
    return max(known, key=quarter_rank) if known else (labels[-1] if labels else None)
