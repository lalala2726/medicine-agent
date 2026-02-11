from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

"""
Excel 领域实体模型集合。

该文件用于沉淀 Excel 底层能力可复用的数据结构。
"""


@dataclass(slots=True)
class ExcelFieldSummary:
    """
    单个字段（列）的概述信息。

    该类对应一个“标题单元格”，例如 A1 / B1。
    """

    column_index: int
    column_letter: str
    header_cell: str
    header_name: str
    data_start_cell: str
    non_empty_data_rows: int
    empty_data_rows: int

    def to_dict(self) -> dict[str, Any]:
        """转为适合大模型理解的字段字典。"""
        return {
            "column_index": self.column_index,
            "column_letter": self.column_letter,
            "header_cell": self.header_cell,
            "header_name": self.header_name,
            "data_start_cell": self.data_start_cell,
            "metrics": {
                "non_empty_data_rows": self.non_empty_data_rows,
                "empty_data_rows": self.empty_data_rows,
            },
        }


@dataclass(slots=True)
class ExcelSheetSummary:
    """
    工作表级别概述信息。

    该类聚合：
    - 表级概览（标题数量、数据行数量）；
    - 字段列表（A1/B1...）；
    - 按标题单元格索引的快速访问映射。
    """

    sheet_name: str
    header_row_number: int
    data_start_row_number: int
    headers: list[str] = field(default_factory=list)
    fields: list[ExcelFieldSummary] = field(default_factory=list)
    data_rows_total: int = 0
    data_rows_with_any_value: int = 0

    def to_llm_dict(self) -> dict[str, Any]:
        """
        转为分层字典，方便大模型直接消费。
        """
        fields_list = [item.to_dict() for item in self.fields]
        field_index = {
            item.header_cell: {
                "name": item.header_name,
                "non_empty_data_rows": item.non_empty_data_rows,
                "empty_data_rows": item.empty_data_rows,
                "data_start_cell": item.data_start_cell,
            }
            for item in self.fields
        }
        return {
            "sheet": {
                "name": self.sheet_name,
                "header_row": self.header_row_number,
                "data_start_row": self.data_start_row_number,
            },
            "overview": {
                "header_count": len(self.headers),
                "data_rows_total": self.data_rows_total,
                "data_rows_with_any_value": self.data_rows_with_any_value,
            },
            "headers": self.headers,
            "fields": fields_list,
            "field_index": field_index,
        }


@dataclass(slots=True)
class ExcelCellLocation:
    """
    Excel 单元格位置信息。
    """

    sheet_name: str
    row_number: int
    column_index: int
    column_letter: str
    cell_address: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sheet_name": self.sheet_name,
            "row_number": self.row_number,
            "column_index": self.column_index,
            "column_letter": self.column_letter,
            "cell_address": self.cell_address,
        }


@dataclass(slots=True)
class ExcelKeywordMatch:
    """
    关键词命中记录。
    """

    location: ExcelCellLocation
    cell_value: str
    matched_keyword: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location.to_dict(),
            "cell_value": self.cell_value,
            "matched_keyword": self.matched_keyword,
        }


@dataclass(slots=True)
class ExcelSheetMatchGroup:
    """
    单个工作表下的关键词命中分组。
    """

    sheet_name: str
    matches: list[ExcelKeywordMatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sheet_name": self.sheet_name,
            "match_count": len(self.matches),
            "matches": [item.to_dict() for item in self.matches],
        }


@dataclass(slots=True)
class ExcelKeywordSearchResult:
    """
    Excel 关键词搜索结果。
    """

    keyword: str
    scanned_sheet_count: int
    scanned_cell_count: int
    total_match_count: int
    returned_match_count: int
    truncated: bool
    by_sheet: list[ExcelSheetMatchGroup] = field(default_factory=list)
    matches: list[ExcelKeywordMatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "summary": {
                "scanned_sheet_count": self.scanned_sheet_count,
                "scanned_cell_count": self.scanned_cell_count,
                "total_match_count": self.total_match_count,
                "returned_match_count": self.returned_match_count,
                "truncated": self.truncated,
            },
            "by_sheet": [item.to_dict() for item in self.by_sheet],
            "matches": [item.to_dict() for item in self.matches],
        }
