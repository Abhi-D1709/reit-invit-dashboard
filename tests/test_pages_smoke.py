"""Smoke test: every page loads without raising.

Needs the network (Google Sheets, the data branch), so it only runs when RUN_SMOKE=1.
It exists to catch environment breakage such as a dependency upgrade removing an API
(this is how the pandas 3 `applymap` crash on the Governance page would have been caught).
"""
import glob
import os

import pytest

pytestmark = pytest.mark.skipif(not os.getenv("RUN_SMOKE"), reason="set RUN_SMOKE=1 to run (needs network)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = sorted(glob.glob(os.path.join(ROOT, "pages", "*.py")))


@pytest.mark.parametrize("page", PAGES, ids=[os.path.basename(p) for p in PAGES])
def test_page_loads(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(page, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
