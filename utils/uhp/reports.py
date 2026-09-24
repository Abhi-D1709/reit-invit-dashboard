"""Excel report generation for a single Unit Holding Pattern filing."""

from io import BytesIO

import pandas as pd

from utils.uhp import ownership_reports
from utils.uhp import sebi_format
from utils.uhp.xbrl_parser import ParsedUHP

HEADER_FILL = "0F3D68"
SUBTOTAL_FILL = "EAF0F5"
TOTAL_FILL = "D3E3EE"


def _write_df(writer, df: pd.DataFrame, sheet_name: str, title: str, workbook, kind_col: str | None = None):
    df_out = df.drop(columns=["_kind"]) if kind_col and "_kind" in df.columns else df
    df_out.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)
    ws = writer.sheets[sheet_name]

    title_fmt = workbook.add_format({"bold": True, "font_size": 13})
    ws.write(0, 0, title, title_fmt)

    header_fmt = workbook.add_format(
        {"bold": True, "bg_color": "#" + HEADER_FILL, "font_color": "white", "border": 1, "text_wrap": True}
    )
    for col_idx, col_name in enumerate(df_out.columns):
        ws.write(2, col_idx, col_name, header_fmt)

    subtotal_fmt = workbook.add_format({"bold": True, "bg_color": "#" + SUBTOTAL_FILL})
    total_fmt = workbook.add_format({"bold": True, "bg_color": "#" + TOTAL_FILL})
    num_fmt = workbook.add_format({"num_format": "#,##0.00"})

    if kind_col and "_kind" in df.columns:
        for row_idx, kind in enumerate(df["_kind"]):
            excel_row = row_idx + 3
            fmt = None
            if kind in ("subtotal",):
                fmt = subtotal_fmt
            elif kind in ("total", "grand_total"):
                fmt = total_fmt
            if fmt is not None:
                for col_idx in range(len(df_out.columns)):
                    val = df_out.iloc[row_idx, col_idx]
                    ws.write(excel_row, col_idx, val if pd.notna(val) else None, fmt)

    ws.set_column(0, 0, 46)
    if len(df_out.columns) > 1:
        ws.set_column(1, len(df_out.columns) - 1, 20, num_fmt)


def build_excel_report(
    parsed: ParsedUHP,
    header_info: dict,
    as_on_date: str,
) -> bytes:
    table_i = sebi_format.build_table_i(parsed)
    any_other = sebi_format.build_any_other_breakup(parsed)
    other_unitholders = sebi_format.build_other_unitholders_table(parsed)
    manager_shareholders = sebi_format.build_manager_shareholders_table(parsed)
    directors_kmp = sebi_format.build_directors_kmp_table(parsed)
    dom_for = ownership_reports.build_domestic_foreign_report(parsed)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Summary / header sheet
        ws_name = "Summary"
        header_df = pd.DataFrame(list(header_info.items()), columns=["Field", "Value"])
        header_df.loc[len(header_df)] = ["As On Date", as_on_date]
        header_df.to_excel(writer, sheet_name=ws_name, startrow=2, index=False, header=False)
        ws = writer.sheets[ws_name]
        title_fmt = workbook.add_format({"bold": True, "font_size": 14})
        ws.write(0, 0, "Unit Holding Pattern - Filing Summary", title_fmt)
        bold_fmt = workbook.add_format({"bold": True})
        for i in range(len(header_df)):
            ws.write(i + 2, 0, header_df.iloc[i, 0], bold_fmt)
        ws.set_column(0, 0, 34)
        ws.set_column(1, 1, 50)

        _write_df(
            writer,
            table_i,
            "Table I - Unit Holding Pattern",
            f"Table I: Statement showing Unit Holding Pattern as on {as_on_date}",
            workbook,
            kind_col="_kind",
        )

        _write_df(
            writer,
            dom_for["table"],
            "Domestic vs Foreign",
            f"Domestic vs Foreign Ownership as on {as_on_date}",
            workbook,
        )
        if not dom_for["audit"].empty:
            _write_df(
                writer,
                dom_for["audit"][["Bucket", "Nature of 'Any Other'", "No. of units held", "Classified as"]],
                "DomForeign-Audit",
                "Domestic/Foreign classification audit trail for 'Any Other' categories",
                workbook,
            )

        for title, df in any_other.items():
            sheet = title.replace("Break-up of 'Any Other' - ", "Any Other - ")[:31]
            _write_df(writer, df, sheet, title, workbook)

        if not other_unitholders.empty:
            _write_df(
                writer,
                other_unitholders,
                "TableII-UnitHolders",
                "Table II(A): Unit holders (other than sponsor) and % of unit holding",
                workbook,
            )

        if not manager_shareholders.empty:
            _write_df(
                writer,
                manager_shareholders,
                "TableII-ManagerHolders",
                "Table II(B): Unit holding of shareholders/partners of the Manager/Investment Manager",
                workbook,
            )

        if not directors_kmp.empty:
            _write_df(
                writer,
                directors_kmp,
                "TableIII-DirectorsKMP",
                "Table III: Details of Directors/KMPs of the Manager/Investment Manager",
                workbook,
            )

    return buffer.getvalue()


def build_trend_csv(trend_df: pd.DataFrame) -> bytes:
    return trend_df.to_csv(index=False).encode("utf-8")
