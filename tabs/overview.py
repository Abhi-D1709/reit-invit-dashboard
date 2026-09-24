# tabs/overview.py
"""Overview: one row per REIT, one column per compliance area, worst verdict in each cell.

Each cell comes from that area's `summary_results(entity)`, which applies the same checks as the
area's own page to the latest reported period. A cell never hides a problem: FAIL beats REVIEW
beats "No data" beats PASS. Areas that cannot be loaded show "No data" with the reason, not a pass.
"""
from __future__ import annotations

import html
from dataclasses import replace

import pandas as pd
import streamlit as st

from utils import rules, status
from utils.common import DEFAULT_REIT_BORR_URL, DEFAULT_REIT_SPON_URL, ENT_COL
from utils.status import CheckResult, Status

from tabs import borrowings, investment, ndcf, sponsor_holding, valuation

# (area name, adapter, url path of its page)
AREAS = [
    (borrowings.AREA, borrowings.summary_results, "Borrowings"),
    (sponsor_holding.AREA, sponsor_holding.summary_results, "Sponsor_Holding"),
    (investment.AREA, investment.summary_results, "Investment"),
    (ndcf.AREA, ndcf.summary_results, "NDCF"),
    (valuation.AREA, valuation.summary_results, "Valuation"),
]
NOT_COVERED = "Governance, Related party transactions, Unit holding pattern, Fund raising and Trading"


def _by_rule_confidence(result: CheckResult) -> CheckResult:
    """A failure resting on a rule that has not been fully checked against the regulation is shown as
    Review, not Fail: the overview must not put a hard red mark on a rule we are not sure of."""
    rule = rules.RULES.get(result.rule_key)
    if result.status is Status.FAIL and rule is not None and rule.status in (rules.PARTIAL, rules.UNVERIFIED):
        return replace(result, status=Status.REVIEW, message=f"{result.message} (this rule is {rule.status}; see Rules reference)")
    return result


def run_area(entity: str, area: str, adapter) -> list[CheckResult]:
    """Run one area's checks; a failure to load or compute becomes 'No data' with the reason."""
    try:
        results = adapter(entity)
        return [_by_rule_confidence(r) for r in results] or [CheckResult(area, Status.NO_DATA, "No checks could be run", area)]
    except Exception as e:  # a broken sheet must not take the whole overview down
        return [CheckResult(area, Status.NO_DATA, f"Could not run these checks: {type(e).__name__}: {e}", area)]


def collect(entities: list[str]) -> dict[str, dict[str, list[CheckResult]]]:
    return {e: {area: run_area(e, area, fn) for area, fn, _ in AREAS} for e in entities}


def area_status(results: list[CheckResult]) -> Status:
    return status.worst(r.status for r in results)


def counts(collected: dict) -> dict[Status, int]:
    """How many individual checks ended in each status."""
    out = {s: 0 for s in Status}
    for by_area in collected.values():
        for results in by_area.values():
            for r in results:
                out[r.status] += 1
    return out


def details_frame(by_area: dict[str, list[CheckResult]]) -> pd.DataFrame:
    """Every check for one entity, most severe first."""
    order = {s: i for i, s in enumerate([Status.FAIL, Status.REVIEW, Status.NO_DATA, Status.PASS, Status.NA])}
    rows = []
    for results in by_area.values():
        for r in results:
            rule = rules.RULES.get(r.rule_key)
            rows.append({
                "Area": r.area, "Check": r.check, "Result": r.text, "Detail": r.message,
                "Basis": rule.source if rule else "", "_o": order[r.status],
            })
    return pd.DataFrame(rows).sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)


def matrix_html(collected: dict) -> str:
    """The overview grid as an HTML table (header cells scoped for screen readers, wraps to scroll on a phone)."""
    head = "".join(f'<th scope="col" role="columnheader">{html.escape(area)}</th>' for area, _, _ in AREAS)
    body = []
    for entity, by_area in collected.items():
        cells = "".join(
            f'<td role="cell" data-label="{html.escape(area)}">{status.badge_html(area_status(by_area[area]))}</td>' for area, _, _ in AREAS
        )
        body.append(f'<tr role="row"><th scope="row" role="rowheader">{html.escape(entity)}</th>{cells}</tr>')
    # explicit roles keep the table semantics when the CSS turns rows into cards on a phone
    return (
        '<div class="sc-wrap"><table class="scorecard" role="table">'
        '<caption class="sr-only">Compliance status by REIT and area, latest reported period</caption>'
        f'<thead role="rowgroup"><tr role="row"><th scope="col" role="columnheader">REIT</th>{head}</tr></thead>'
        f'<tbody role="rowgroup">{"".join(body)}</tbody></table></div>'
    )


def stats_html(n_reits: int, n: dict) -> str:
    """The headline counts as a compact strip (two per row on a phone)."""
    items = [
        ("REITs covered", n_reits), ("Checks failing", n[Status.FAIL]), ("Need review", n[Status.REVIEW]), ("No data", n[Status.NO_DATA]),
    ]
    cells = "".join(f'<div class="stat"><div class="stat-n">{v}</div><div class="stat-l">{html.escape(label)}</div></div>' for label, v in items)
    return f'<div class="stat-strip">{cells}</div>'


def _entities() -> list[str]:
    names: set[str] = set()
    for load, url in ((borrowings.load_borrowings_url, DEFAULT_REIT_BORR_URL), (sponsor_holding._load_sponsor_df, DEFAULT_REIT_SPON_URL)):
        try:
            names.update(load(url)[ENT_COL].dropna().astype(str).unique())
        except Exception:
            pass
    return sorted(names)


def render() -> None:
    st.title("Compliance overview")
    st.markdown("The latest reported period for each REIT, checked against the rules on the Rules reference page.")

    entities = _entities()
    if not entities:
        st.warning("Could not load the list of REITs, so the overview cannot be built. Open one of the pages to see the error.")
        return
    with st.spinner("Running the checks…"):
        collected = collect(entities)

    n = counts(collected)
    st.markdown(stats_html(len(entities), n), unsafe_allow_html=True)

    st.markdown(matrix_html(collected), unsafe_allow_html=True)
    st.caption(
        "Each cell is the worst result among that area's checks: "
        + "  ".join(status.badge_html(s) for s in (Status.FAIL, Status.REVIEW, Status.NO_DATA, Status.PASS))
        + ". Open a REIT below for every check, its detail and its regulatory basis.",
        unsafe_allow_html=True,
    )

    st.subheader("Details by REIT")
    for entity, by_area in collected.items():
        table = details_frame(by_area)
        worst = status.worst(r.status for results in by_area.values() for r in results)
        fails = sum(r.status is Status.FAIL for results in by_area.values() for r in results)
        reviews = sum(r.status is Status.REVIEW for results in by_area.values() for r in results)
        summary = " · ".join(x for x in (f"{fails} failing" if fails else "", f"{reviews} to review" if reviews else "") if x) or status.LABEL[worst]
        with st.expander(f"{entity} — {summary}", expanded=worst is Status.FAIL):
            st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Open a page")
    from utils import navigation  # imported here: navigation imports this module

    links = st.columns(len(AREAS))
    for col, (area, _, path) in zip(links, AREAS):
        if path in navigation.PAGE_BY_PATH:
            col.page_link(navigation.PAGE_BY_PATH[path], label=area)

    st.info(
        f"Not in the overview yet: {NOT_COVERED}, and InvITs. Their pages work as before.",
        icon=":material/info:",
    )
