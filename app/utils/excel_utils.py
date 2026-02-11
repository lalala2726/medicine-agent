from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Union

import pandas as pd

from app.schemas.excel_entities import (
    ExcelCellLocation,
    ExcelFieldSummary,
    ExcelKeywordMatch,
    ExcelKeywordSearchResult,
    ExcelSheetMatchGroup,
    ExcelSheetSummary,
)

# 默认读取标题时最多扫描 50 列（A ~ AX）。
DEFAULT_MAX_HEADER_COLUMNS = 50
# 安全上限：最多只允许扫描 200 列，避免一次读取超大范围导致不必要的内存开销。
MAX_HEADER_COLUMNS_LIMIT = 200


def _normalize_max_columns(max_columns: int) -> int:
    """
    规范化最大读取列数。

    规则：
    1) 必须是正整数；
    2) 当调用方传入超过 200 的值时，自动收敛到 200。
    """
    if max_columns <= 0:
        raise ValueError("max_columns 必须大于 0。")
    return min(max_columns, MAX_HEADER_COLUMNS_LIMIT)


def _to_excel_column_letter(column_index_1_based: int) -> str:
    """
    将 1-based 列索引转换为 Excel 列字母。

    例如：
    - 1 -> A
    - 26 -> Z
    - 27 -> AA
    """
    if column_index_1_based <= 0:
        raise ValueError("column_index_1_based 必须大于 0。")

    column_letter = ""
    value = column_index_1_based
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        column_letter = chr(65 + remainder) + column_letter
    return column_letter


def _build_cell_address(row_number: int, column_index: int) -> str:
    """
    构建 Excel 单元格地址（如 B12）。
    """
    if row_number <= 0:
        raise ValueError("row_number 必须大于 0。")
    return f"{_to_excel_column_letter(column_index)}{row_number}"


def _normalize_keyword(keyword: str) -> str:
    """
    规范化关键词。
    """
    normalized = keyword.strip()
    if not normalized:
        raise ValueError("keyword 不能为空。")
    return normalized


def _is_non_empty_cell(value: Any) -> bool:
    """
    判断单元格是否为“有效非空值”。

    规则：
    - NaN / None 视为无值；
    - 字符串去除首尾空格后为空，视为无值；
    - 其他值（包括 0）视为有值。
    """
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _cell_matches_keyword(
        value: Any,
        keyword: str,
        case_sensitive: bool = False,
) -> bool:
    """
    判断单元格值是否命中关键词。

    默认采用不区分大小写的“包含匹配”。
    """
    if not _is_non_empty_cell(value):
        return False

    text = str(value)
    if case_sensitive:
        return keyword in text
    return keyword.lower() in text.lower()


def _resolve_sheet_name(
        sheet_name: Union[int, str],
        available_sheet_names: list[str],
) -> str:
    """
    将调用方传入的工作表参数解析为真实工作表名称。
    """
    if isinstance(sheet_name, int):
        if sheet_name < 0 or sheet_name >= len(available_sheet_names):
            raise ValueError(
                f"工作表索引越界: {sheet_name}，可用范围: 0 ~ {len(available_sheet_names) - 1}"
            )
        return available_sheet_names[sheet_name]

    if sheet_name not in available_sheet_names:
        raise ValueError(
            f"Worksheet named '{sheet_name}' not found. 可用工作表: {available_sheet_names}"
        )
    return sheet_name


def build_excel_sheet_summary(
        file_path: Union[str, Path],
        sheet_name: Union[int, str] = 0,
        *,
        header_row: int = 0,
        max_columns: int = DEFAULT_MAX_HEADER_COLUMNS,
) -> ExcelSheetSummary:
    """
    构建 Excel 工作表概述对象（类）。
    """
    if header_row < 0:
        raise ValueError("header_row 不能小于 0。")

    normalized_max_columns = _normalize_max_columns(max_columns)
    resolved_path = Path(file_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {resolved_path}")

    excel_file = pd.ExcelFile(resolved_path)
    actual_sheet_name = _resolve_sheet_name(sheet_name, excel_file.sheet_names)
    header_row_number = header_row + 1
    data_start_row_number = header_row + 2

    # 先只读取到标题行，快速定位连续标题范围。
    header_frame = pd.read_excel(
        resolved_path,
        sheet_name=actual_sheet_name,
        header=None,
        nrows=header_row + 1,
    )

    if header_frame.empty or header_row >= len(header_frame.index):
        return ExcelSheetSummary(
            sheet_name=actual_sheet_name,
            header_row_number=header_row_number,
            data_start_row_number=data_start_row_number,
        )

    raw_header_values = header_frame.iloc[header_row].tolist()[:normalized_max_columns]
    detected_headers: list[tuple[int, str]] = []
    for column_index_0_based, value in enumerate(raw_header_values):
        if not _is_non_empty_cell(value):
            break
        detected_headers.append((column_index_0_based, str(value).strip()))

    if not detected_headers:
        return ExcelSheetSummary(
            sheet_name=actual_sheet_name,
            header_row_number=header_row_number,
            data_start_row_number=data_start_row_number,
        )

    selected_column_indexes = [column_index for column_index, _ in detected_headers]
    data_frame = pd.read_excel(
        resolved_path,
        sheet_name=actual_sheet_name,
        header=None,
        usecols=selected_column_indexes,
    )

    if len(data_frame.index) > header_row + 1:
        data_section = data_frame.iloc[header_row + 1:]
    else:
        data_section = data_frame.iloc[0:0]
    data_rows_total = len(data_section.index)

    fields: list[ExcelFieldSummary] = []
    non_empty_mask_columns: list[pd.Series] = []
    for selected_col_position, (
        column_index_0_based,
        header_title,
    ) in enumerate(detected_headers):
        column_index_1_based = column_index_0_based + 1
        column_letter = _to_excel_column_letter(column_index_1_based)
        header_cell = f"{column_letter}{header_row_number}"
        data_start_cell = f"{column_letter}{data_start_row_number}"

        if data_rows_total > 0:
            column_series = data_section.iloc[:, selected_col_position]
            non_empty_mask = column_series.map(_is_non_empty_cell)
            non_empty_rows = int(non_empty_mask.sum())
            empty_rows = int(data_rows_total - non_empty_rows)
            non_empty_mask_columns.append(non_empty_mask)
        else:
            non_empty_rows = 0
            empty_rows = 0

        fields.append(
            ExcelFieldSummary(
                column_index=column_index_1_based,
                column_letter=column_letter,
                header_cell=header_cell,
                header_name=header_title,
                data_start_cell=data_start_cell,
                non_empty_data_rows=non_empty_rows,
                empty_data_rows=empty_rows,
            )
        )

    if non_empty_mask_columns:
        # 数据区中“任意列有值”的行数，反映有效数据规模。
        non_empty_any_column = pd.concat(non_empty_mask_columns, axis=1).any(axis=1)
        data_rows_with_any_value = int(non_empty_any_column.sum())
    else:
        data_rows_with_any_value = 0

    return ExcelSheetSummary(
        sheet_name=actual_sheet_name,
        header_row_number=header_row_number,
        data_start_row_number=data_start_row_number,
        headers=[header_title for _, header_title in detected_headers],
        fields=fields,
        data_rows_total=data_rows_total,
        data_rows_with_any_value=data_rows_with_any_value,
    )


def read_excel_header_summary(
        file_path: Union[str, Path],
        sheet_name: Union[int, str] = 0,
        *,
        header_row: int = 0,
        max_columns: int = DEFAULT_MAX_HEADER_COLUMNS,
) -> dict[str, Any]:
    """
    读取 Excel 标题并返回结构化概述信息（字典）。

    主要用于“给 AI 读取系统导出的 Excel 字段结构”场景，返回信息包括：
    - 标题列表；
    - 每个标题对应的单元格（如 A1、B1）；
    - 每个标题列在标题行下方的非空行数；
    - 表级的行列概览（如检测到标题个数、数据起始行等）。

    Args:
        file_path: Excel 文件路径（支持 `str` 或 `Path`）。
        sheet_name: 工作表名或索引，默认读取第一个工作表（0）。
        header_row: 标题所在行（0-based），默认 0 表示 Excel 第 1 行。
        max_columns: 最多扫描的列数，默认 50，最大生效值 200。

    Returns:
        概述字典，核心字段如下：
        - sheet: 工作表层信息（名称、标题行、数据起始行）。
        - overview: 表级统计（标题数量、数据行总数、任意列有值的行数）。
        - headers: 标题名数组。
        - fields: 字段详情数组（A1/B1... 及对应统计）。
        - field_index: 以 A1/B1... 为 key 的字段映射，便于快速访问。
    """
    summary = build_excel_sheet_summary(
        file_path=file_path,
        sheet_name=sheet_name,
        header_row=header_row,
        max_columns=max_columns,
    )
    return summary.to_llm_dict()


def read_excel_headers(
        file_path: Union[str, Path],
        sheet_name: Union[int, str] = 0,
        *,
        header_row: int = 0,
        max_columns: int = DEFAULT_MAX_HEADER_COLUMNS,
) -> list[str]:
    """
    读取 Excel 指定工作表中“标题行”的列标题。

    该方法用于处理“系统导出 Excel 的首行字段名”场景：
    - 默认从第 1 行（`header_row=0`）开始读取；
    - 从 A 列开始向右读取，直到遇到空值/空字符串为止；
    - 默认最多读取 50 列，可通过 `max_columns` 调整，最大不会超过 200 列。

    例如：
    - A1 ~ M1 有标题时，返回 M1 前所有标题；
    - A1 ~ N1 有标题时，返回 N1 前所有标题；
    - 若 A1、B1 有值，C1 为空，则只返回 A1、B1。

    Args:
        file_path: Excel 文件路径（支持 `str` 或 `Path`）。
        sheet_name: 工作表名或索引，默认读取第一个工作表（0）。
        header_row: 标题所在行（0-based），默认 0 表示 Excel 第 1 行。
        max_columns: 最多扫描的列数，默认 50，最大生效值 200。

    Returns:
        标题列表（按从左到右顺序）。

    Raises:
        ValueError:
            - `header_row` 小于 0；
            - `max_columns` 小于等于 0。
        FileNotFoundError:
            - 指定 Excel 文件不存在。
    """
    summary = read_excel_header_summary(
        file_path=file_path,
        sheet_name=sheet_name,
        header_row=header_row,
        max_columns=max_columns,
    )
    return summary["headers"]


def list_excel_sheets(
        file_path: Union[str, Path]
) -> dict[str, Any]:
    """
    读取 Excel 的工作表列表（底层能力，不依赖 AI 工具装饰器）。

    返回结构说明：
    - sheet_count: 工作表总数
    - sheet_names: 工作表名称列表（按原始顺序）
    - sheets: 包含 index + name 的数组，便于模型按“索引或名称”引用
    - default_sheet: 默认工作表（通常为第一个）
    """
    resolved_path = Path(file_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {resolved_path}")

    excel_file = pd.ExcelFile(resolved_path)
    sheet_names = excel_file.sheet_names
    sheets = [{"index": index, "name": name} for index, name in enumerate(sheet_names)]

    default_sheet = sheets[0] if sheets else None
    return {
        "sheet_count": len(sheet_names),
        "sheet_names": sheet_names,
        "sheets": sheets,
        "default_sheet": default_sheet,
    }


def search_excel_keyword(
        file_path: Union[str, Path],
        keyword: str,
        *,
        sheet_name: Union[int, str, None] = None,
        max_results: int = 200,
) -> dict[str, Any]:
    """
    在 Excel 中按关键词搜索单元格，返回分层与平铺结构。

    默认行为：
    - 全表（所有 sheet）扫描；
    - 包含匹配；
    - 不区分大小写；
    - 最多返回 200 条命中结果，达到上限后提前停止扫描。
    """
    normalized_keyword = _normalize_keyword(keyword)
    normalized_max_results = _normalize_max_columns(max_results)
    resolved_path = Path(file_path).expanduser()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {resolved_path}")

    excel_file = pd.ExcelFile(resolved_path)
    available_sheet_names = excel_file.sheet_names
    target_sheet_names = (
        [_resolve_sheet_name(sheet_name, available_sheet_names)]
        if sheet_name is not None
        else available_sheet_names
    )

    flat_matches: list[ExcelKeywordMatch] = []
    grouped_matches: "OrderedDict[str, list[ExcelKeywordMatch]]" = OrderedDict()

    scanned_sheet_count = 0
    scanned_cell_count = 0
    total_match_count = 0
    truncated = False
    should_stop = False

    for current_sheet_name in target_sheet_names:
        if should_stop:
            break

        scanned_sheet_count += 1
        dataframe = pd.read_excel(
            resolved_path,
            sheet_name=current_sheet_name,
            header=None,
        )
        values = dataframe.to_numpy(dtype=object)

        for row_index_0_based, row_values in enumerate(values):
            if should_stop:
                break
            row_number = row_index_0_based + 1

            for column_index_0_based, cell_value in enumerate(row_values):
                scanned_cell_count += 1

                if not _cell_matches_keyword(
                        value=cell_value,
                        keyword=normalized_keyword,
                        case_sensitive=False,
                ):
                    continue

                total_match_count += 1

                if len(flat_matches) >= normalized_max_results:
                    truncated = True
                    should_stop = True
                    break

                column_index = column_index_0_based + 1
                location = ExcelCellLocation(
                    sheet_name=current_sheet_name,
                    row_number=row_number,
                    column_index=column_index,
                    column_letter=_to_excel_column_letter(column_index),
                    cell_address=_build_cell_address(row_number, column_index),
                )
                match = ExcelKeywordMatch(
                    location=location,
                    cell_value=str(cell_value),
                    matched_keyword=normalized_keyword,
                )
                flat_matches.append(match)
                grouped_matches.setdefault(current_sheet_name, []).append(match)

                if len(flat_matches) >= normalized_max_results:
                    truncated = True
                    should_stop = True
                    break

    by_sheet = [
        ExcelSheetMatchGroup(sheet_name=group_sheet_name, matches=matches)
        for group_sheet_name, matches in grouped_matches.items()
    ]

    result = ExcelKeywordSearchResult(
        keyword=normalized_keyword,
        scanned_sheet_count=scanned_sheet_count,
        scanned_cell_count=scanned_cell_count,
        total_match_count=total_match_count,
        returned_match_count=len(flat_matches),
        truncated=truncated,
        by_sheet=by_sheet,
        matches=flat_matches,
    )
    return result.to_dict()
