"""Smoke test: every page loads without raising.

Needs the network (Google Sheets, the data branch), so it only runs when RUN_SMOKE=1.
It exists to catch environment breakage such as a dependency upgrade removing an API
(this is how the pandas 3 `applymap` crash on the Governance page would have been caught).
"""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.getenv("RUN_SMOKE"), reason="set RUN_SMOKE=1 to run (needs network)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = [
    "basic_details", "fundraising", "borrowings", "trading", "ndcf", "sponsor_holding", "governance",
    "valuation", "related_party", "investment", "unit_holding", "rules_reference", "overview",
]


def _run_page(module_name):
    import importlib

    import streamlit as st

    st.set_page_config(layout="wide")
    from utils.common import inject_global_css

    inject_global_css()
    importlib.import_module(f"tabs.{module_name}").render()


@pytest.mark.parametrize("module", MODULES)
def test_page_loads(module):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_run_page, args=(module,), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_entrypoint_boots_on_the_overview():
    """app.py builds the navigation and runs the default page (the overview) inside it."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("compliance overview" in t.value.lower() for t in at.title)
