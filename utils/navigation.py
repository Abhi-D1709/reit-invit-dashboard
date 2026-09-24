# utils/navigation.py
"""The page tree: which pages exist, how they are grouped in the sidebar, and their URLs.

Replaces the twelve one-line wrapper files that used to live in pages/. URL paths keep the old
names (for example /Borrowings) so existing bookmarks still work.
"""
from __future__ import annotations

import streamlit as st

PAGE_BY_PATH: dict[str, "st.Page"] = {}


def build() -> dict[str, list]:
    """Sections of pages for st.navigation. Also fills PAGE_BY_PATH so pages can link to each other."""
    from tabs import (
        basic_details, borrowings, fundraising, governance, investment, ndcf, overview,
        related_party, rules_reference, sponsor_holding, trading, unit_holding, valuation,
    )

    def page(fn, title, icon, path, default=False):
        p = st.Page(fn, title=title, icon=f":material/{icon}:", url_path=path, default=default)
        PAGE_BY_PATH[path] = p
        return p

    return {
        "Summary": [
            page(overview.render, "Overview", "dashboard", "Overview", default=True),
        ],
        "Compliance checks": [
            page(borrowings.render, "Borrowings", "account_balance", "Borrowings"),
            page(ndcf.render, "NDCF distributions", "payments", "NDCF"),
            page(sponsor_holding.render, "Sponsor & public holding", "pie_chart", "Sponsor_Holding"),
            page(investment.render, "Investment conditions", "domain", "Investment"),
            page(governance.render, "Governance", "gavel", "Governance"),
            page(valuation.render, "Valuation", "request_quote", "Valuation"),
            page(related_party.render, "Related party transactions", "handshake", "Related_Party"),
        ],
        "Market & ownership": [
            page(trading.render, "Trading", "candlestick_chart", "Trading"),
            page(unit_holding.render, "Unit holding pattern", "groups", "Unit_Holding_Pattern"),
        ],
        "Reference": [
            page(basic_details.render, "Basic details", "badge", "Basic_Details"),
            page(fundraising.render, "Fund raising", "savings", "Fund_Raising"),
            page(rules_reference.render, "Rules reference", "menu_book", "Rules_Reference"),
        ],
    }
