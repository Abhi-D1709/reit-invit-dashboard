# tabs/rules_reference.py
"""Rules reference: every threshold the dashboard applies, where it comes from, and how far it was checked.

Read-only view of utils/rules.py, the single place these numbers are defined.
"""
import pandas as pd
import streamlit as st

from utils import rules

STATUS_ICON = {
    rules.VERIFIED: "✅ verified",
    rules.PARTIAL: "🟡 partly verified",
    rules.HOUSE_RULE: "🔷 house rule",
    rules.UNVERIFIED: "⚪ not verified",
}


def rules_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Area": r.area,
                "Rule": r.name,
                "Value": rules.display_value(r),
                "Applies to": r.applies_to,
                "Source": r.source,
                "Status": STATUS_ICON[r.status],
                "Checked by the app": "yes" if r.enforced else "no",
                "Note": r.note,
            }
            for r in rules.RULES.values()
        ]
    )


def render():
    st.header("Rules reference")
    st.markdown(
        "Every threshold the dashboard applies is defined in one file (`utils/rules.py`). "
        "This table shows each value, where it comes from and how far it has been checked."
    )
    df = rules_frame()

    c1, c2, c3, c4 = st.columns(4)
    counts = df["Status"].value_counts()
    c1.metric("Verified against the regulation", int(counts.get(STATUS_ICON[rules.VERIFIED], 0)))
    c2.metric("Partly verified", int(counts.get(STATUS_ICON[rules.PARTIAL], 0)))
    c3.metric("House rules", int(counts.get(STATUS_ICON[rules.HOUSE_RULE], 0)))
    c4.metric("Not verified", int(counts.get(STATUS_ICON[rules.UNVERIFIED], 0)))

    with st.sidebar:
        st.subheader("Filter")
        area = st.selectbox("Area", ["All"] + sorted(df["Area"].unique()), key="rules_area")
        status = st.selectbox("Status", ["All"] + list(STATUS_ICON.values()), key="rules_status")
        not_checked = st.checkbox("Only rules the app does not check yet", key="rules_unchecked")

    view = df
    if area != "All":
        view = view[view["Area"] == area]
    if status != "All":
        view = view[view["Status"] == status]
    if not_checked:
        view = view[view["Checked by the app"] == "no"]
    st.dataframe(view, width="stretch", hide_index=True)

    st.caption(
        f"Regulation text reviewed: {rules.REGS_CHECKED} ([PDF]({rules.REGS_URL})), and SEBI circular "
        "SEBI/HO/DDHS/PoD2/P/CIR/2023/106 of 27 Jun 2023. Later amendments and the InvIT Regulations have not been checked. "
        "This is a reading of the official text, not legal advice."
    )
    st.info(
        "**House rule** = an alert chosen by the dashboard owner, not a regulatory limit. "
        "**Not verified** = carried over from earlier code and not yet checked against a source.",
        icon=":material/info:",
    )
