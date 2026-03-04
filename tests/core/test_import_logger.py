"""import_logger 模块单元测试。"""

from app.core.mq.import_logger import ImportStage, import_log


def test_import_log_info_stage_does_not_raise() -> None:
    """
    测试目的：验证 INFO 级别阶段（如 DOWNLOAD_DONE）不抛异常即可。
    预期结果：函数正常返回 None。
    """
    result = import_log(
        ImportStage.DOWNLOAD_DONE,
        "uuid-1",
        filename="a.pdf",
        size=1024,
    )
    assert result is None


def test_import_log_error_stage_does_not_raise() -> None:
    """
    测试目的：验证 ERROR 级别阶段（如 FAILED）不抛异常。
    预期结果：函数正常返回 None。
    """
    result = import_log(
        ImportStage.FAILED,
        "uuid-2",
        error="download timeout",
    )
    assert result is None


def test_import_log_warning_stage_does_not_raise() -> None:
    """
    测试目的：验证 WARNING 级别阶段（如 RETRY_SCHEDULED）不抛异常。
    预期结果：函数正常返回 None。
    """
    result = import_log(
        ImportStage.RETRY_SCHEDULED,
        "uuid-3",
        attempt=2,
        delay_seconds=30,
    )
    assert result is None


def test_import_log_no_metrics() -> None:
    """
    测试目的：验证不传附加指标时日志函数仍可正常运行。
    预期结果：函数正常返回 None。
    """
    result = import_log(ImportStage.TASK_RECEIVED, "uuid-4")
    assert result is None


def test_import_log_default_task_uuid() -> None:
    """
    测试目的：验证 task_uuid 使用默认值 "-" 时正常运行。
    预期结果：函数正常返回 None。
    """
    result = import_log(ImportStage.CONSUMER_CONNECTED)
    assert result is None


def test_import_stage_values_are_strings() -> None:
    """
    测试目的：验证 ImportStage 枚举值都是字符串类型。
    预期结果：所有枚举值均为 str。
    """
    for stage in ImportStage:
        assert isinstance(stage.value, str)
        assert stage.value == stage.value.lower()


def test_import_log_captures_output(capsys) -> None:
    """
    测试目的：验证日志输出包含关键信息（task_uuid、stage、metrics）。
    预期结果：stderr 中包含 task_uuid 和 stage 标记。
    """
    import sys

    from loguru import logger

    # 添加临时 sink 到 stderr 以配合 capsys
    sink_id = logger.add(sys.stderr, format="{message}", level="DEBUG")
    try:
        import_log(
            ImportStage.DOWNLOAD_DONE,
            "test-uuid-cap",
            filename="demo.pdf",
            size=9999,
        )
        captured = capsys.readouterr()
        assert "test-uuid-cap" in captured.err
        assert "download_done" in captured.err
        assert "filename=demo.pdf" in captured.err
        assert "size=9999" in captured.err
    finally:
        logger.remove(sink_id)
