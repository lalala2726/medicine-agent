"""切片重建 MQ 模型测试。"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.core.mq.contracts.document.chunk_rebuild_models import (
    ChunkRebuildResultStage,
    KnowledgeChunkRebuildCommandMessage,
    KnowledgeChunkRebuildResultMessage,
)


def test_chunk_rebuild_result_stage_enum_values() -> None:
    """验证切片重建结果阶段枚举值保持稳定。"""
    assert ChunkRebuildResultStage.STARTED.value == "STARTED"
    assert ChunkRebuildResultStage.COMPLETED.value == "COMPLETED"
    assert ChunkRebuildResultStage.FAILED.value == "FAILED"


def test_chunk_rebuild_command_message_serialization() -> None:
    """验证切片重建命令消息序列化包含必要字段。"""
    command = KnowledgeChunkRebuildCommandMessage(
        task_uuid="task-1",
        knowledge_name="demo_kb",
        document_id=7,
        vector_id=101,
        version=3,
        content="  new chunk content  ",
        embedding_model="text-embedding-v4",
        created_at=datetime.now(timezone.utc),
    )

    payload = command.model_dump()
    assert payload["message_type"] == "knowledge_chunk_rebuild_command"
    assert payload["vector_id"] == 101
    assert payload["version"] == 3
    assert payload["content"] == "new chunk content"

    serialized = command.model_dump_json().encode("utf-8")
    assert isinstance(serialized, bytes)
    assert b"knowledge_chunk_rebuild_command" in serialized


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_id", 0),
        ("vector_id", 0),
        ("version", 0),
        ("knowledge_name", "1demo"),
        ("content", "   "),
    ],
)
def test_chunk_rebuild_command_message_rejects_invalid_fields(field: str, value) -> None:
    """验证关键字段非法时会触发参数校验错误。"""
    payload = {
        "task_uuid": "task-1",
        "knowledge_name": "demo_kb",
        "document_id": 7,
        "vector_id": 101,
        "version": 3,
        "content": "valid content",
        "embedding_model": "text-embedding-v4",
        "created_at": datetime.now(timezone.utc),
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        KnowledgeChunkRebuildCommandMessage(**payload)


def test_chunk_rebuild_result_message_build_sets_duration() -> None:
    """验证切片重建结果消息构建时会计算耗时并保持结构完整。"""
    start = datetime(2026, 3, 5, 10, 0, 0, tzinfo=timezone.utc)
    occurred = datetime(2026, 3, 5, 10, 0, 2, tzinfo=timezone.utc)

    result = KnowledgeChunkRebuildResultMessage.build(
        task_uuid="task-1",
        version=3,
        stage=ChunkRebuildResultStage.COMPLETED,
        message="切片重建成功",
        knowledge_name="demo_kb",
        document_id=7,
        vector_id=101,
        embedding_model="text-embedding-v4",
        embedding_dim=1024,
        started_at=start,
        occurred_at=occurred,
    )

    assert result.message_type == "knowledge_chunk_rebuild_result"
    assert result.version == 3
    assert result.stage == ChunkRebuildResultStage.COMPLETED
    assert result.embedding_dim == 1024
    assert result.duration_ms == 2000

    serialized = result.to_json_bytes()
    assert isinstance(serialized, bytes)
    assert b"knowledge_chunk_rebuild_result" in serialized
