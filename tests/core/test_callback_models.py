"""CallbackStage 和 KnowledgeImportCallbackPayload 扩展测试。"""

from datetime import datetime, timezone

from app.core.mq.models import CallbackStage, KnowledgeImportCallbackPayload


def _build_payload(
        status: str = "COMPLETED", stage_detail: str | None = None
) -> KnowledgeImportCallbackPayload:
    return KnowledgeImportCallbackPayload.build(
        task_uuid="task-1",
        knowledge_name="demo",
        document_id=7,
        file_url="https://example.com/a.txt",
        status=status,
        message="ok",
        embedding_model="text-embedding-v4",
        embedding_dim=1024,
        chunk_strategy="character",
        chunk_size=500,
        token_size=100,
        chunk_count=3,
        vector_count=3,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        stage_detail=stage_detail,
    )


def test_callback_stage_enum_values() -> None:
    """
    测试目的：验证 CallbackStage 枚举包含所有预期阶段。
    预期结果：四个阶段值完整。
    """
    assert CallbackStage.STARTED.value == "STARTED"
    assert CallbackStage.PROCESSING.value == "PROCESSING"
    assert CallbackStage.COMPLETED.value == "COMPLETED"
    assert CallbackStage.FAILED.value == "FAILED"


def test_callback_stage_from_legacy_success() -> None:
    """
    测试目的：验证旧版 SUCCESS 状态正确映射到 COMPLETED。
    预期结果：from_legacy("SUCCESS") 返回 CallbackStage.COMPLETED。
    """
    assert CallbackStage.from_legacy("SUCCESS") == CallbackStage.COMPLETED


def test_callback_stage_from_legacy_failed() -> None:
    """
    测试目的：验证旧版 FAILED 状态保持不变。
    预期结果：from_legacy("FAILED") 返回 CallbackStage.FAILED。
    """
    assert CallbackStage.from_legacy("FAILED") == CallbackStage.FAILED


def test_callback_stage_from_legacy_new_value() -> None:
    """
    测试目的：验证新枚举值（STARTED / PROCESSING）可以直接通过 from_legacy 传递。
    预期结果：from_legacy("STARTED") 返回 CallbackStage.STARTED。
    """
    assert CallbackStage.from_legacy("STARTED") == CallbackStage.STARTED
    assert CallbackStage.from_legacy("PROCESSING") == CallbackStage.PROCESSING


def test_payload_to_callback_body_excludes_none_stage_detail() -> None:
    """
    测试目的：验证 to_callback_body 在 stage_detail 为 None 时不包含该字段。
    预期结果：返回字典中不存在 stage_detail 键。
    """
    payload = _build_payload(stage_detail=None)
    body = payload.to_callback_body()
    assert "stage_detail" not in body


def test_payload_to_callback_body_includes_stage_detail() -> None:
    """
    测试目的：验证 to_callback_body 在 stage_detail 有值时包含该字段。
    预期结果：返回字典包含正确的 stage_detail。
    """
    payload = _build_payload(stage_detail="downloading")
    body = payload.to_callback_body()
    assert body["stage_detail"] == "downloading"


def test_payload_to_callback_body_contains_required_fields() -> None:
    """
    测试目的：验证 to_callback_body 包含所有必要字段。
    预期结果：必要字段全部存在。
    """
    payload = _build_payload()
    body = payload.to_callback_body()
    required_keys = {
        "task_uuid", "knowledge_name", "document_id", "file_url",
        "status", "message", "embedding_model", "embedding_dim",
        "chunk_strategy", "chunk_size", "token_size",
        "chunk_count", "vector_count", "started_at", "finished_at", "duration_ms",
    }
    assert required_keys.issubset(body.keys())


def test_payload_dates_serialized_as_iso8601() -> None:
    """
    测试目的：验证日期字段在 to_callback_body 中序列化为 ISO8601 字符串。
    预期结果：started_at 和 finished_at 均为字符串格式。
    """
    payload = _build_payload()
    body = payload.to_callback_body()
    assert isinstance(body["started_at"], str)
    assert isinstance(body["finished_at"], str)
    assert "T" in body["started_at"]
