# tabs/unit_holding.py
"""Unit Holding Pattern Analysis - REITs & InvITs (SEBI-prescribed format).

Data: NSE corporate-unit-holdings master feed, plus BSE for BSE-only trusts
(XBRL filings). Data layer lives in utils/uhp/.
"""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.uhp import ownership_reports, peer_benchmark, reports, sebi_format
from utils.uhp import sources as filing_source
from utils.uhp.xbrl_parser import parse_uhp_xbrl, to_number

CHART_FONT = "IBM Plex Sans, sans-serif"


def _parse_date(d: str):
    try:
        return datetime.strptime(d, "%d-%b-%Y")
    except (ValueError, TypeError):
        return None


@st.cache_data(ttl=3600)
def get_master_df(index: str) -> tuple[pd.DataFrame, list[str]]:
    records, problems = filing_source.fetch_master(index)
    df = pd.DataFrame(records)
    if df.empty:
        return df, problems
    df["asOnDateParsed"] = df["asOnDate"].apply(_parse_date)
    df["entityKey"] = df["ndsSymbol"] + " — " + df["secLname"]
    df["publicHoldingPer"] = pd.to_numeric(df["publicHoldingPer"], errors="coerce")
    df["sponsorGroupPer"] = pd.to_numeric(df["sponsorGroupPer"], errors="coerce")
    return df.sort_values("asOnDateParsed", ascending=False), problems


def style_table_i(df: pd.DataFrame):
    pct_cols = [c for c in df.columns if c.startswith("As a %")]
    num_cols = [c for c in df.columns if c.startswith("No. of")]

    display_df = df.drop(columns=["_kind"]).copy()
    for c in num_cols:
        display_df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
    for c in pct_cols:
        display_df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}%")

    def row_style(row):
        kind = df.loc[row.name, "_kind"]
        if kind == "grand_total":
            return ["background-color: #0F3D68; color: white; font-weight: 600"] * len(row)
        if kind == "total":
            return ["background-color: #D3E3EE; font-weight: 600"] * len(row)
        if kind == "subtotal":
            return ["background-color: #EAF0F5; font-weight: 600"] * len(row)
        if kind == "header":
            return ["font-weight: 700; color: #0F3D68"] * len(row)
        return [""] * len(row)

    return display_df.style.apply(row_style, axis=1)


def render_donut(sponsor_pct: float, public_pct: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Sponsor & Sponsor Group", "Public"],
                values=[sponsor_pct, public_pct],
                hole=0.6,
                marker=dict(colors=["#0F3D68", "#B08D3E"]),
                textfont=dict(family=CHART_FONT, size=13),
            )
        ]
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(family=CHART_FONT)),
        font=dict(family=CHART_FONT, color="#1A2433"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_category_bar(breakdown_df: pd.DataFrame) -> go.Figure:
    df = breakdown_df[breakdown_df["% of Total Units"] > 0].sort_values("% of Total Units")
    fig = go.Figure(
        go.Bar(
            x=df["% of Total Units"],
            y=df["Category"],
            orientation="h",
            marker=dict(color="#0F3D68"),
            text=df["% of Total Units"].map(lambda v: f"{v:.2f}%"),
            textposition="outside",
            textfont=dict(family=CHART_FONT),
        )
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=40),
        height=max(300, 30 * len(df)),
        xaxis_title="% of total outstanding units",
        yaxis_title=None,
        font=dict(family=CHART_FONT, color="#1A2433"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#EAF0F5"),
    )
    return fig


def render_domestic_foreign_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Domestic",
            x=df["Segment"],
            y=df["Domestic %"],
            marker_color="#0F3D68",
            text=df["Domestic %"].map(lambda v: f"{v:.2f}%"),
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Foreign",
            x=df["Segment"],
            y=df["Foreign %"],
            marker_color="#B08D3E",
            text=df["Foreign %"].map(lambda v: f"{v:.2f}%"),
            textposition="inside",
        )
    )
    if df["Unclassified %"].sum() > 0:
        fig.add_trace(
            go.Bar(
                name="Unclassified",
                x=df["Segment"],
                y=df["Unclassified %"],
                marker_color="#B0B8C1",
                text=df["Unclassified %"].map(lambda v: f"{v:.2f}%" if v > 0 else ""),
                textposition="inside",
            )
        )
    fig.update_layout(
        barmode="stack",
        margin=dict(t=10, b=10, l=10, r=10),
        height=380,
        yaxis_title="% of segment's total units",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(family=CHART_FONT)),
        font=dict(family=CHART_FONT, color="#1A2433"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#EAF0F5"),
    )
    return fig


def render_trend(entity_df: pd.DataFrame, threshold: float | None = None) -> go.Figure:
    df = entity_df.sort_values("asOnDateParsed")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["asOnDateParsed"],
            y=df["sponsorGroupPer"],
            mode="lines+markers",
            name="Sponsor & Sponsor Group %",
            line=dict(color="#0F3D68", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["asOnDateParsed"],
            y=df["publicHoldingPer"],
            mode="lines+markers",
            name="Public Holding %",
            line=dict(color="#B08D3E", width=2.5),
        )
    )
    if threshold is not None:
        breach = df[df["publicHoldingPer"] < threshold]
        if not breach.empty:
            fig.add_trace(
                go.Scatter(
                    x=breach["asOnDateParsed"],
                    y=breach["publicHoldingPer"],
                    mode="markers",
                    name="Below threshold",
                    marker=dict(color="#A23B3B", size=11, symbol="x"),
                )
            )
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="#A23B3B",
            annotation_text=f"Threshold: {threshold:.0f}%",
            annotation_position="top left",
        )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=380,
        yaxis_title="% of total outstanding units",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(family=CHART_FONT)),
        font=dict(family=CHART_FONT, color="#1A2433"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#EAF0F5"),
        yaxis=dict(gridcolor="#EAF0F5"),
    )
    return fig


def render_peer_bar(df: pd.DataFrame, threshold: float) -> go.Figure:
    df = df.sort_values("publicHoldingPer")
    colors = ["#A23B3B" if v < threshold else "#0F3D68" for v in df["publicHoldingPer"]]
    fig = go.Figure(
        go.Bar(
            x=df["publicHoldingPer"],
            y=df["entityKey"],
            orientation="h",
            marker=dict(color=colors),
            text=df["publicHoldingPer"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(family=CHART_FONT),
        )
    )
    fig.add_vline(
        x=threshold,
        line_dash="dot",
        line_color="#A23B3B",
        annotation_text=f"Threshold: {threshold:.0f}%",
        annotation_position="top",
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=60),
        height=max(320, 30 * len(df)),
        xaxis_title="Public holding, % of total outstanding units (latest filing)",
        yaxis_title=None,
        font=dict(family=CHART_FONT, color="#1A2433"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#EAF0F5"),
    )
    return fig


def render_peer_benchmarking(index_label: str, master_df: pd.DataFrame):
    st.subheader(f"Peer benchmarking — {index_label}", icon=":material/balance:")
    st.caption("Latest available filing for each entity, compared across the sector.")

    latest_df = peer_benchmark.latest_filing_per_entity(master_df)

    threshold = st.slider(
        "Minimum public holding threshold (%)",
        min_value=5,
        max_value=75,
        value=25,
        step=1,
        key="uhp_peer_threshold",
        help=(
            "User-defined reference line, not an asserted regulatory figure. SEBI's prescribed minimum "
            "public unitholding for REITs/InvITs can vary by trust size and listing vintage — set the "
            "threshold applicable to the trusts you are reviewing."
        ),
    )
    below = latest_df[latest_df["publicHoldingPer"] < threshold]

    with st.container(horizontal=True):
        st.metric("Entities compared", f"{len(latest_df)}", border=True)
        st.metric("Average public holding", f"{latest_df['publicHoldingPer'].mean():.2f}%", border=True)
        st.metric("Below threshold", f"{len(below)}", border=True)

    with st.container(border=True):
        st.markdown("**Public holding % by entity (latest filing)**")
        st.plotly_chart(render_peer_bar(latest_df, threshold), width="stretch")

    if not below.empty:
        st.warning(
            f"{len(below)} entit{'y is' if len(below) == 1 else 'ies are'} below the {threshold:.0f}% "
            "public holding threshold as of their latest filing.",
            icon=":material/warning:",
        )

    display_df = latest_df[["entityKey", "asOnDate", "sponsorGroupPer", "publicHoldingPer"]].rename(
        columns={
            "entityKey": "Entity",
            "asOnDate": "As on date",
            "sponsorGroupPer": "Sponsor & sponsor group %",
            "publicHoldingPer": "Public holding %",
        }
    )

    if st.button(
        "Load extended metrics (domestic/foreign, lock-in)",
        icon=":material/travel_explore:",
        help="Fetches and parses each entity's latest XBRL filing — slower on first run.",
    ):
        extended = peer_benchmark.build_extended_peer_metrics(latest_df)
        if not extended.empty:
            extended = extended.rename(columns={"entityKey": "Entity"})
            display_df = display_df.merge(extended, on="Entity", how="left")

    st.dataframe(
        display_df.style.format(
            {c: "{:,.2f}%" for c in display_df.columns if c not in ("Entity", "As on date")}
        ),
        width="stretch",
        hide_index=True,
        height=min(700, 40 + 36 * len(display_df)),
    )


def render():
    st.title("Unit Holding Pattern Analysis", icon=":material/account_balance:")
    st.caption("REITs & InvITs listed on NSE and BSE — SEBI-prescribed Unit Holding Pattern format")

    with st.sidebar:
        st.subheader("Select filing", icon=":material/tune:")
        index_label = st.segmented_control(
            "Instrument type", ["InvITs", "REITs"], default="InvITs", label_visibility="collapsed", key="uhp_type"
        )
        index_label = index_label or "InvITs"
        index = "invits" if index_label == "InvITs" else "reits"

        master_df, problems = get_master_df(index)
        for problem in problems:
            st.warning(problem, icon=":material/warning:")

        if master_df.empty:
            st.error("No filings returned for this index. NSE may be blocking the request; try Refresh in a minute.")
            st.stop()

        view_mode = st.segmented_control(
            "View", ["Single entity", "Peer benchmarking"], default="Single entity", key="uhp_view"
        )
        view_mode = view_mode or "Single entity"

        entity = as_on_date = entity_df = None
        if view_mode == "Single entity":
            entities = sorted(master_df["entityKey"].unique())
            entity = st.selectbox("Entity", entities, key="uhp_entity")

            entity_df = master_df[master_df["entityKey"] == entity]
            dates = entity_df.sort_values("asOnDateParsed", ascending=False)["asOnDate"].tolist()
            as_on_date = st.selectbox("As on date", dates, key="uhp_asof")

            st.caption(f":material/history: {len(entity_df)} filing(s) available for this entity")
            source = entity_df["source"].iloc[0]
            st.caption(
                ":material/database: Source: NSE"
                if source == "NSE"
                else ":material/database: Source: BSE (not listed on NSE)"
            )

        if st.button("Refresh filing list", icon=":material/refresh:", width="stretch"):
            get_master_df.clear()
            filing_source.clear_caches()
            st.rerun()

    if view_mode == "Peer benchmarking":
        render_peer_benchmarking(index_label, master_df)
        return

    record = entity_df[entity_df["asOnDate"] == as_on_date].iloc[0]
    xml_text = filing_source.fetch_xbrl(record["xbrlFilePath"])
    parsed = parse_uhp_xbrl(xml_text)
    header_info = sebi_format.build_header_info(parsed)

    sponsor_pct = to_number(parsed.get("UnitHoldingOfSponsorAndSponsorGroupI", "AsAPercentageOfTotalOutStandingUnits"))
    public_pct = to_number(parsed.get("PublicHoldingI", "AsAPercentageOfTotalOutStandingUnits"))
    locked_in_pct = to_number(parsed.get("TotalUnitsOutstandingI", "AsAPercentageOfTotalUnitsMandatorilyHeld"))
    pledged_pct = to_number(parsed.get("TotalUnitsOutstandingI", "AsAPercentageOfTotalNumberOfUnitsPledgedOrOtherwiseEncumbered"))
    total_units = to_number(parsed.get("TotalUnitsOutstandingI", "NumberOfUnitsHeld"))
    dom_for_report = ownership_reports.build_domestic_foreign_report(parsed)

    # Filing identity banner
    with st.container(border=True):
        c1, c2 = st.columns([2.2, 1])
        with c1:
            st.markdown(f"##### {header_info.get('Name of the Entity') or record['secLname']}")
            badges = (
                f":blue-badge[NSE: {header_info.get('NSE Symbol') or '—'}] "
                f":gray-badge[BSE: {header_info.get('BSE Scrip Code') or '—'}] "
                f":orange-badge[{header_info.get('Type of Report') or '—'}]"
            )
            st.markdown(badges)
            st.caption(f"SEBI Registration No.: {header_info.get('SEBI Registration Number') or '—'}")
        with c2:
            st.markdown(f"**As on date**  \n{as_on_date}")
            fy = f"{header_info.get('Financial Year Start') or '—'} to {header_info.get('Financial Year End') or '—'}"
            st.caption(f"FY: {fy}")

    tab_overview, tab_domfor, tab_table1, tab_table2, tab_table3, tab_trend, tab_report = st.tabs(
        [
            ":material/dashboard: Overview",
            ":material/public: Domestic vs foreign",
            ":material/table_chart: Table I — holding pattern",
            ":material/groups: Table II — holders",
            ":material/badge: Table III — directors/KMP",
            ":material/trending_up: Trend & compliance",
            ":material/description: Reports",
        ]
    )

    with tab_overview:
        with st.container(horizontal=True):
            st.metric("Total units outstanding", f"{total_units:,.0f}", border=True)
            st.metric("Sponsor & sponsor group holding", f"{sponsor_pct:.2f}%", border=True)
            st.metric("Public holding", f"{public_pct:.2f}%", border=True)
            st.metric("Units mandatorily held / locked-in", f"{locked_in_pct:.2f}%", border=True)
            st.metric("Units pledged / encumbered", f"{pledged_pct:.2f}%", border=True)

        col1, col2 = st.columns([1, 1.4])
        with col1:
            with st.container(border=True):
                st.markdown("**Sponsor vs public**")
                st.plotly_chart(render_donut(sponsor_pct, public_pct), width="stretch")
        with col2:
            with st.container(border=True):
                st.markdown("**Category-wise holding**")
                breakdown_df = sebi_format.build_category_breakdown(parsed)
                if not breakdown_df.empty:
                    st.plotly_chart(render_category_bar(breakdown_df), width="stretch")
                else:
                    st.caption("No non-zero category breakdown reported in this filing.")

    with tab_domfor:
        st.subheader("Domestic vs foreign ownership", icon=":material/public:")
        st.caption(f"As on {as_on_date} — overall, and split by sponsor group vs. public")

        dom_for_table = dom_for_report["table"]
        overall_row = dom_for_table[dom_for_table["Segment"] == "Overall"].iloc[0]
        with st.container(horizontal=True):
            st.metric("Overall domestic holding", f"{overall_row['Domestic %']:.2f}%", border=True)
            st.metric("Overall foreign holding", f"{overall_row['Foreign %']:.2f}%", border=True)
            if overall_row["Unclassified %"] > 0:
                st.metric("Unclassified", f"{overall_row['Unclassified %']:.2f}%", border=True)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            with st.container(border=True):
                st.markdown("**Domestic vs foreign, by segment**")
                st.plotly_chart(render_domestic_foreign_bar(dom_for_table), width="stretch")
        with col2:
            with st.container(border=True):
                st.markdown("**Summary table**")
                display_cols = ["Segment", "Domestic %", "Foreign %", "Unclassified %"]
                st.dataframe(
                    dom_for_table[display_cols].style.format(
                        {c: "{:,.2f}%" for c in display_cols if c != "Segment"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.caption("Figures are % of that segment's own total units (Sponsor, Public, or Overall).")

        with st.expander("Units (absolute) and classification methodology", icon=":material/info:"):
            units_cols = ["Segment", "Domestic units", "Foreign units", "Unclassified units", "Total units"]
            st.dataframe(
                dom_for_table[units_cols].style.format({c: "{:,.0f}" for c in units_cols if c != "Segment"}),
                width="stretch",
                hide_index=True,
            )
            st.markdown(
                "**Methodology:** Sponsor & Sponsor Group is split into Domestic/Foreign directly from "
                "SEBI's Table I categories (Indian vs. Foreign sub-totals). Public holding has no such "
                "direct split in the SEBI taxonomy, so it is derived: Foreign Portfolio Investors, Foreign "
                "Venture Capital Investors and Non-Resident Indians are treated as foreign; Mutual Funds, "
                "domestic Financial Institutions/Banks, Insurance Companies, Provident/Pension Funds, "
                "resident Individuals, NBFCs, Trusts, Clearing Members and Bodies Corporate are treated as "
                "domestic. Each 'Any Other (specify)' category is classified line-by-line from its disclosed "
                "nature (e.g. 'Foreign Company', 'Non-Resident...'); a lump sum with no such break-up "
                "disclosed is marked **Unclassified** rather than guessed."
            )
            if not dom_for_report["audit"].empty:
                st.markdown("**'Any Other' break-up classification audit:**")
                audit_df = dom_for_report["audit"][["Bucket", "Nature of 'Any Other'", "No. of units held", "Classified as"]]
                st.dataframe(
                    audit_df.style.format({"No. of units held": "{:,.0f}"}),
                    width="stretch",
                    hide_index=True,
                )

    with tab_table1:
        st.subheader("Table I: Statement showing unit holding pattern", icon=":material/table_chart:")
        st.caption(f"As on {as_on_date}")
        table_i = sebi_format.build_table_i(parsed)
        st.dataframe(style_table_i(table_i), width="stretch", height=760, hide_index=True)

        any_other = sebi_format.build_any_other_breakup(parsed)
        if any_other:
            st.subheader("Break-up of 'Any Other' categories", icon=":material/list_alt:")
            for title, df in any_other.items():
                with st.expander(title):
                    st.dataframe(
                        df.style.format(
                            {c: "{:,.2f}" for c in df.columns if c not in ("S.No.", "Nature of 'Any Other'")}
                        ),
                        width="stretch",
                        hide_index=True,
                    )

    with tab_table2:
        st.subheader("Table II(A): Unit holders other than sponsor", icon=":material/groups:")
        df_holders = sebi_format.build_other_unitholders_table(parsed)
        if df_holders.empty:
            st.caption("Not disclosed in this filing.")
        else:
            st.dataframe(df_holders, width="stretch", hide_index=True)

        st.subheader(
            "Table II(B): Unit holding of shareholders/partners of the Manager/Investment Manager",
            icon=":material/business_center:",
        )
        df_mgr = sebi_format.build_manager_shareholders_table(parsed)
        if df_mgr.empty:
            st.caption("Not disclosed in this filing.")
        else:
            st.dataframe(df_mgr, width="stretch", hide_index=True)

    with tab_table3:
        st.subheader("Table III: Directors / KMPs of the Manager / Investment Manager", icon=":material/badge:")
        df_dir = sebi_format.build_directors_kmp_table(parsed)
        if df_dir.empty:
            st.caption("Not disclosed in this filing.")
        else:
            for _, row in df_dir.iterrows():
                with st.expander(f"{row['Name']} — {row['Designation'] or ''}"):
                    st.markdown(f"**Action:** {row['Appointment/Resignation/Removal'] or '—'}")
                    st.markdown(f"**Date:** {row['Date'] or '—'}")
                    if row["Brief Profile"]:
                        st.write(row["Brief Profile"])

    with tab_trend:
        st.subheader(f"Sponsor vs public holding trend — {header_info.get('Name of the Entity') or entity}", icon=":material/trending_up:")

        threshold = st.slider(
            "Minimum public holding threshold (%)",
            min_value=5,
            max_value=75,
            value=25,
            step=1,
            key="uhp_entity_threshold",
            help=(
                "User-defined reference line, not an asserted regulatory figure. SEBI's prescribed minimum "
                "public unitholding for REITs/InvITs can vary by trust size and listing vintage — set the "
                "threshold applicable to this trust."
            ),
        )
        breach_df = entity_df[entity_df["publicHoldingPer"] < threshold].sort_values("asOnDate", ascending=False)

        if len(entity_df) < 2:
            st.caption("Only one filing available for this entity — no trend to show.")
        else:
            with st.container(border=True):
                st.plotly_chart(render_trend(entity_df, threshold=threshold), width="stretch")

        if breach_df.empty:
            st.success(
                f"Public holding has stayed at or above {threshold:.0f}% in every filing on record for this entity.",
                icon=":material/check_circle:",
            )
        else:
            st.warning(
                f"Public holding fell below {threshold:.0f}% in {len(breach_df)} of {len(entity_df)} filing(s) on record.",
                icon=":material/warning:",
            )

        st.dataframe(
            entity_df[["asOnDate", "sponsorGroupPer", "publicHoldingPer", "submissionDate"]]
            .sort_values("asOnDate", ascending=False)
            .rename(
                columns={
                    "asOnDate": "As on date",
                    "sponsorGroupPer": "Sponsor & sponsor group %",
                    "publicHoldingPer": "Public holding %",
                    "submissionDate": "Submission date",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_report:
        st.subheader("Export report", icon=":material/description:")
        with st.container(border=True):
            st.markdown(f"**Filing report** — {header_info.get('Name of the Entity') or entity}, as on {as_on_date}")
            excel_bytes = reports.build_excel_report(parsed, header_info, as_on_date)
            st.download_button(
                "Download Excel report (SEBI format)",
                data=excel_bytes,
                file_name=f"UHP_{record['ndsSymbol']}_{as_on_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/download:",
            )

        with st.container(border=True):
            st.markdown(f"**Historical trend data** — {header_info.get('Name of the Entity') or entity}")
            trend_csv = reports.build_trend_csv(
                entity_df[["asOnDate", "sponsorGroupPer", "publicHoldingPer", "submissionDate", "xbrlFilePath"]]
            )
            st.download_button(
                "Download trend data (CSV)",
                data=trend_csv,
                file_name=f"UHP_trend_{record['ndsSymbol']}.csv",
                mime="text/csv",
                icon=":material/download:",
            )

        with st.expander("Raw XBRL source", icon=":material/code:"):
            st.link_button("Open source XBRL filing", record["xbrlFilePath"], icon=":material/open_in_new:")

