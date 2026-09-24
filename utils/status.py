# utils/status.py
"""One vocabulary for the outcome of a compliance check, shared by every page.

A verdict is never carried by colour alone: each status has a glyph and a word ("✔ Pass",
"✖ Fail", "▲ Review", "? No data", "– n/a"), so it reads the same in a table, on a screen reader,
for a colour-blind user, and in light or dark mode. Colour is added on top (see `badge_html`).

  PASS     the check ran and the rule is met
  FAIL     the check ran and the rule is not met
  REVIEW   needs a person to look (an alert that is not a plain breach, or a warning about the data)
  NO_DATA  the check could not run because an input is missing or unreadable ("insufficient data")
  NA       the rule does not apply to this row (for example the other timeline of a rule that changed)

Missing data is NO_DATA, never PASS and never FAIL.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import pandas as pd


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"
    NO_DATA = "no_data"
    NA = "na"


GLYPH = {Status.PASS: "✔", Status.FAIL: "✖", Status.REVIEW: "▲", Status.NO_DATA: "?", Status.NA: "–"}
LABEL = {Status.PASS: "Pass", Status.FAIL: "Fail", Status.REVIEW: "Review", Status.NO_DATA: "No data", Status.NA: "n/a"}
# Streamlit alert function that presents each status, and its icon
ALERT = {Status.PASS: "success", Status.FAIL: "error", Status.REVIEW: "warning", Status.NO_DATA: "info", Status.NA: "info"}
ICON = {
    Status.PASS: ":material/check_circle:",
    Status.FAIL: ":material/cancel:",
    Status.REVIEW: ":material/warning:",
    Status.NO_DATA: ":material/help:",
    Status.NA: ":material/remove:",
}
# worst first: a failure outranks a warning, which outranks missing data, which outranks a pass
_SEVERITY = [Status.FAIL, Status.REVIEW, Status.NO_DATA, Status.PASS, Status.NA]


def tag(status: Status, text: Optional[str] = None) -> str:
    """Glyph plus words, e.g. tag(FAIL) -> '✖ Fail', tag(PASS, '12.5% (no alert)') -> '✔ 12.5% (no alert)'."""
    return f"{GLYPH[status]} {text if text else LABEL[status]}"


def tagged(status: Status, detail: str) -> str:
    """Glyph, word and detail: tagged(FAIL, '75.00% (below 80%)') -> '✖ Fail · 75.00% (below 80%)'."""
    return f"{GLYPH[status]} {LABEL[status]} · {detail}"


def from_bool(value) -> Status:
    """True -> PASS, False -> FAIL, None / NaN / <NA> -> NO_DATA."""
    if value is None or (not isinstance(value, bool) and pd.isna(value)):
        return Status.NO_DATA
    return Status.PASS if bool(value) else Status.FAIL


PASS_TAG, FAIL_TAG, REVIEW_TAG, NO_DATA_TAG, NA_TAG = (tag(s) for s in (Status.PASS, Status.FAIL, Status.REVIEW, Status.NO_DATA, Status.NA))

_FROM_ALERT = {"success": Status.PASS, "error": Status.FAIL, "warning": Status.REVIEW, "info": Status.NO_DATA}


def from_alert(level: str) -> Status:
    """The status behind a Streamlit alert name ("success", "error", "warning", "info")."""
    return _FROM_ALERT[level]


def of_text(text) -> Optional[Status]:
    """The status a tag()-formatted cell starts with, or None if it doesn't start with one."""
    s = str(text).lstrip()
    for status, glyph in GLYPH.items():
        if s.startswith(glyph):
            return status
    return None


def has_status(series: pd.Series, status: Status) -> bool:
    """True if any cell of a text column starts with this status's glyph."""
    return bool(series.astype(str).str.lstrip().str.startswith(GLYPH[status]).any())


def worst(statuses: Iterable[Status]) -> Status:
    """The most severe of several statuses (NA if there are none)."""
    present = set(statuses)
    return next((s for s in _SEVERITY if s in present), Status.NA)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check for one entity: what was checked, the verdict, and why."""

    check: str
    status: Status
    message: str = ""
    area: str = ""
    rule_key: str = ""  # key in utils/rules.py, when the check applies a threshold from there

    @property
    def text(self) -> str:
        return tag(self.status)


def show(status: Status, message: str) -> None:
    """Show a message in the Streamlit alert box for this status (with an icon, not just a colour)."""
    import streamlit as st

    getattr(st, ALERT[status])(message, icon=ICON[status])


def badge_html(status: Status, text: Optional[str] = None) -> str:
    """A pill: glyph and word, tinted by status. The text keeps the page's own text colour, so it
    stays readable in both themes; the tint is decoration only. Styles come from inject_global_css."""
    label = html.escape(text if text else LABEL[status])
    return f'<span class="badge badge-{status.value}"><span aria-hidden="true">{GLYPH[status]}</span> {label}</span>'
