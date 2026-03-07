"""知识库导入 MQ 模型测试。"""

from datetime import datetime, timezone

from app.core.mq.contracts.document.import_models import (
    ImportResultStage,
    KnowledgeImportCommandMessage,
    KnowledgeImportResultMessage,
)
from app.rag.chunking import ChunkStrategyType


def test_import_result_stage_enum_values() -> None:
    """验证结果阶段枚举值保持稳定。"""
    assert ImportResultStage.STARTED.value == "STARTED"
    assert ImportResultStage.PROCESSING.value == "PROCESSING"
    assert ImportResultStage.COMPLETED.value == "COMPLETED"
    assert ImportResultStage.FAILED.value == "FAILED"

def test_command_message_serialization() -> None:
    """验证命令消息序列化包含必要字段。"""
    command = KnowledgeImportCommandMessage(
        task_uuid="task-1",
        biz_key="demo:7",
        version=3,
        knowledge_name="demo",
        document_id=7,
        file_url="https://example.com/a.txt",
        embedding_model="text-embedding-v4",
        chunk_strategy=ChunkStrategyType.CHARACTER,
        chunk_size=500,
        token_size=100,
        created_at=datetime.now(timezone.utc),
    )

    payload = command.model_dump()
    assert payload["message_type"] == "knowledge_import_command"
    assert payload["biz_key"] == "demo:7"
    assert payload["version"] == 3

    serialized = command.to_json_bytes()
    assert isinstance(serialized, bytes)
    assert b"knowledge_import_command" in serialized


def test_result_message_build_sets_duration() -> None:
    """验证结果消息构建时会计算耗时并保持结构完整。"""
    start = datetime(2026, 3, 5, 10, 0, 0, tzinfo=timezone.utc)
    occurred = datetime(2026, 3, 5, 10, 0, 2, tzinfo=timezone.utc)

    result = KnowledgeImportResultMessage.build(
        task_uuid="task-1",
        biz_key="demo:7",
        version=3,
        stage=ImportResultStage.PROCESSING,
        message="任务处理中",
        knowledge_name="demo",
        document_id=7,
        file_url="https://example.com/a.txt",
        embedding_model="text-embedding-v4",
        started_at=start,
        occurred_at=occurred,
    )

    assert result.message_type == "knowledge_import_result"
    assert result.stage == ImportResultStage.PROCESSING
    payload = result.model_dump(exclude_none=True)
    assert set(payload.keys()) == {
        "message_type",
        "task_uuid",
        "biz_key",
        "version",
        "stage",
        "message",
        "knowledge_name",
        "document_id",
        "file_url",
        "chunk_count",
        "vector_count",
        "embedding_model",
        "embedding_dim",
        "occurred_at",
        "duration_ms",
    }
    assert result.duration_ms == 2000

    serialized = result.to_json_bytes()
    assert isinstance(serialized, bytes)
    assert b"knowledge_import_result" in serialized
