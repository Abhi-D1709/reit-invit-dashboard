# utils/constants.py
"""Dependency-free constants shared by the app and the data jobs.

Must not import streamlit: jobs/ runs in CI without it.
"""

# Entities master (Type of Entity, Name, NSE Symbol/Series, BSE Scrip Code, ISIN)
ENTITIES_SHEET_ID = "1g44Lkv3VZU4FDTzrWXKhdxGwrWecZHHrZLmuRMyFHDI"
ENTITIES_SHEET_CSV = f"https://docs.google.com/spreadsheets/d/{ENTITIES_SHEET_ID}/export?format=csv"

# Machine-generated data lives on a separate git branch so that data commits
# don't redeploy the app. Locally it is checked out as a worktree at ./data.
DATA_REPO = "Abhi-D1709/reit-invit-dashboard"
DATA_BRANCH = "data"
DATA_DIR_NAME = "data"

# Earliest date to backfill trading history from. India Grid and IRB InvIT,
# the first listed trusts, began trading in mid-2017.
HISTORY_START = "2017-01-01"
