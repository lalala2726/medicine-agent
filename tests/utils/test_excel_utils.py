from __future__ import annotations

import app.utils.excel_utils as excel_utils_module

# 固定使用本地 Excel 样例文件，便于手工观察打印输出。
TEST_EXCEL_PATH = "/Users/zhangchuang/Downloads/test.xls"


def test_list_excel_sheets_print_result() -> None:
    """
    测试 list_excel_sheets：
    - 读取工作表列表；
    - 打印返回结果，便于确认 sheet 名称与默认 sheet。
    """
    payload = excel_utils_module.list_excel_sheets(TEST_EXCEL_PATH)
    print("\n[list_excel_sheets] result:", flush=True)
    print(payload, flush=True)
    assert isinstance(payload, dict)


def test_read_excel_header_summary_print_result() -> None:
    """
    测试 read_excel_header_summary：
    - 读取结构化概述（sheet/overview/fields/field_index）；
    - 打印结果，便于人工检查 A1/B1 字段名及每列数据行统计。
    """
    summary = excel_utils_module.read_excel_header_summary(
        file_path=TEST_EXCEL_PATH,
        sheet_name=0,
        max_columns=50,
    )
    print("\n[read_excel_header_summary] result:", flush=True)
    print(summary, flush=True)
    assert isinstance(summary, dict)


def test_read_excel_headers_print_result() -> None:
    """
    测试 read_excel_headers：
    - 读取标题列表；
    - 打印标题数组，便于快速核对首行字段顺序。
    """
    headers = excel_utils_module.read_excel_headers(
        file_path=TEST_EXCEL_PATH,
        sheet_name=0,
        max_columns=50,
    )
    print("\n[read_excel_headers] result:", flush=True)
    print(headers, flush=True)
    assert isinstance(headers, list)


def test_search_excel_keyword_print_result() -> None:
    """
    测试 search_excel_keyword：
    - 按关键词全表搜索；
    - 打印分层 + 平铺结果，便于人工核对命中单元格位置。
    """
    result = excel_utils_module.search_excel_keyword(
        file_path=TEST_EXCEL_PATH,
        keyword="课程",
    )
    print("\n[search_excel_keyword] result:", flush=True)
    print(result, flush=True)
    assert isinstance(result, dict)
    assert "summary" in result
    assert "by_sheet" in result
    assert "matches" in result


def test_search_excel_keyword_respects_max_results_print_result() -> None:
    """
    测试 search_excel_keyword 的结果上限：
    - 将 max_results 设置为 5；
    - 打印返回内容并断言返回数量不超过上限。
    """
    result = excel_utils_module.search_excel_keyword(
        file_path=TEST_EXCEL_PATH,
        keyword="课程",
        max_results=5,
    )
    print("\n[search_excel_keyword max_results=5] result:", flush=True)
    print(result, flush=True)

    summary = result["summary"]
    assert summary["returned_match_count"] <= 5
    if summary["returned_match_count"] == 5:
        assert summary["truncated"] is True
