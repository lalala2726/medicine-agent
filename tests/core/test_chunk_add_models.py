"""切片新增 MQ 模型测试。"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.core.mq.contracts.document.chunk_add_models import (
    KnowledgeChunkAddCommandMessage,
    KnowledgeChunkAddResultMessage,
)
from app.core.mq.contracts.document.result_stages import DocumentChunkResultStage


def test_document_chunk_result_stage_enum_values() -> None:
    """验证文档切片链路结果阶段枚举值保持稳定。"""
    assert DocumentChunkResultStage.STARTED.value == "STARTED"
    assert DocumentChunkResultStage.COMPLETED.value == "COMPLETED"
    assert DocumentChunkResultStage.FAILED.value == "FAILED"


def test_chunk_add_command_message_serialization() -> None:
    """验证切片新增命令消息序列化包含必要字段。"""
    command = KnowledgeChunkAddCommandMessage(
        task_uuid="task-1",
        chunk_id=11,
        knowledge_name="demo_kb",
        document_id=7,
        content="  new chunk content  ",
        embedding_model="text-embedding-v4",
        created_at=datetime.now(timezone.utc),
    )

    payload = command.model_dump()
    assert payload["message_type"] == "knowledge_chunk_add_command"
    assert payload["chunk_id"] == 11
    assert payload["content"] == "new chunk content"

    serialized = command.model_dump_json().encode("utf-8")
    assert isinstance(serialized, bytes)
    assert b"knowledge_chunk_add_command" in serialized


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", 0),
        ("document_id", 0),
        ("knowledge_name", "1demo"),
        ("content", "   "),
    ],
)
def test_chunk_add_command_message_rejects_invalid_fields(field: str, value) -> None:
    """验证关键字段非法时会触发参数校验错误。"""
    payload = {
        "task_uuid": "task-1",
        "chunk_id": 11,
        "knowledge_name": "demo_kb",
        "document_id": 7,
        "content": "valid content",
        "embedding_model": "text-embedding-v4",
        "created_at": datetime.now(timezone.utc),
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        KnowledgeChunkAddCommandMessage(**payload)


def test_chunk_add_result_message_build_sets_duration() -> None:
    """验证切片新增结果消息构建时会计算耗时并保持结构完整。"""
    start = datetime(2026, 3, 5, 10, 0, 0, tzinfo=timezone.utc)
    occurred = datetime(2026, 3, 5, 10, 0, 2, tzinfo=timezone.utc)

    result = KnowledgeChunkAddResultMessage.build(
        task_uuid="task-1",
        chunk_id=11,
        stage=DocumentChunkResultStage.COMPLETED,
        message="切片新增成功",
        knowledge_name="demo_kb",
        document_id=7,
        embedding_model="text-embedding-v4",
        vector_id=101,
        chunk_index=3,
        embedding_dim=1024,
        started_at=start,
        occurred_at=occurred,
    )

    assert result.message_type == "knowledge_chunk_add_result"
    assert result.stage == DocumentChunkResultStage.COMPLETED
    assert result.vector_id == 101
    assert result.chunk_index == 3
    assert result.embedding_dim == 1024
    assert result.duration_ms == 2000

    serialized = result.to_json_bytes()
    assert isinstance(serialized, bytes)
    assert b"knowledge_chunk_add_result" in serialized
