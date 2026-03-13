"""Excel 格式化匯出工具模組。

提供將 DataFrame 匯出為具備標題凍結、自動欄寬、背景色標示的 XLSX 檔案功能。
"""

import io
from typing import Optional

import pandas as pd


def export_formatted_xlsx(
    df: pd.DataFrame,
    freeze_row: int = 1,
    header_color: str = "#4472C4",
    font_color: str = "#FFFFFF",
    highlight_cols: Optional[list[str]] = None,
    highlight_color: str = "#FFF2CC",
) -> bytes:
    """將 DataFrame 匯出為格式化的 XLSX 位元組串流。

    Args:
        df: 欲匯出的 DataFrame。
        freeze_row: 凍結列數（從第幾列開始凍結），預設凍結標題列。
        header_color: 標題列背景色（HEX）。
        font_color: 標題列字體色（HEX）。
        highlight_cols: 需要以特殊背景色標示的欄位名稱清單。
        highlight_color: 特殊標示欄位的背景色（HEX）。

    Returns:
        XLSX 檔案的位元組內容。
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="DOE_Experiment")
        workbook = writer.book
        worksheet = writer.sheets["DOE_Experiment"]

        # 標題列格式
        header_fmt = workbook.add_format(
            {
                "bold": True,
                "bg_color": header_color,
                "font_color": font_color,
                "border": 1,
                "text_wrap": True,
                "valign": "vcenter",
                "align": "center",
            }
        )

        # 重點欄位格式
        highlight_fmt = workbook.add_format(
            {
                "bg_color": highlight_color,
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )

        # 一般儲存格格式
        cell_fmt = workbook.add_format(
            {
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )

        # 寫入標題
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, header_fmt)

        # 自動調整欄寬
        for col_idx, col_name in enumerate(df.columns):
            max_len = max(
                len(str(col_name)),
                df[col_name].astype(str).str.len().max() if len(df) > 0 else 0,
            )
            worksheet.set_column(col_idx, col_idx, min(max_len + 4, 30))

        # 套用儲存格格式
        highlight_col_set = set(highlight_cols) if highlight_cols else set()
        for row_idx in range(len(df)):
            for col_idx, col_name in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                fmt = highlight_fmt if col_name in highlight_col_set else cell_fmt
                worksheet.write(row_idx + 1, col_idx, value, fmt)

        # 凍結窗格
        worksheet.freeze_panes(freeze_row, 0)

    return output.getvalue()
