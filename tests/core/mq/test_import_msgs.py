"""导入消息模型单元测试。"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from app.core.mq.models.import_msgs import (
    DEFAULT_COMMAND_CHUNK_SIZE,
    DEFAULT_COMMAND_TOKEN_SIZE,
    KnowledgeImportCommandMessage,
    KnowledgeImportResultMessage,
)
from app.core.mq.models.stages import ImportResultStage
from app.rag.chunking import ChunkStrategyType

# ---- 公共 fixtures -----------------------------------------------------------

_NOW = datetime.now(timezone.utc)

_VALID_COMMAND_DATA = dict(
    task_uuid="uuid-001",
    biz_key="bk-001",
    version=1,
    knowledge_name="test_kb",
    document_id=10,
    file_url="https://example.com/a.pdf",
    embedding_model="text-embedding-v2",
    created_at=_NOW.isoformat(),
)

_VALID_RESULT_KWARGS = dict(
    task_uuid="uuid-001",
    biz_key="bk-001",
    version=1,
    stage=ImportResultStage.COMPLETED,
    message="导入完成",
    knowledge_name="test_kb",
    document_id=10,
    file_url="https://example.com/a.pdf",
    embedding_model="text-embedding-v2",
)


# ---- KnowledgeImportCommandMessage 测试 ------------------------------------


class TestImportCommandMessage:
    """KnowledgeImportCommandMessage 测试。"""

    def test_valid_payload_creates_model(self):
        """测试目的：提供完整有效字段时应成功创建模型实例。
        预期结果：message_type 固定为 knowledge_import_command，各字段值与输入一致。
        """
        msg = KnowledgeImportCommandMessage(**_VALID_COMMAND_DATA)
        assert msg.message_type == "knowledge_import_command"
        assert msg.task_uuid == "uuid-001"
        assert msg.version == 1

    def test_null_chunk_size_falls_back_to_default(self):
        """测试目的：chunk_size 为 None 时应归一为默认值 500。
        预期结果：msg.chunk_size == DEFAULT_COMMAND_CHUNK_SIZE。
        """
        data = {**_VALID_COMMAND_DATA, "chunk_size": None}
        msg = KnowledgeImportCommandMessage(**data)
        assert msg.chunk_size == DEFAULT_COMMAND_CHUNK_SIZE

    def test_null_token_size_falls_back_to_default(self):
        """测试目的：token_size 为 None 时应归一为默认值 100。
        预期结果：msg.token_size == DEFAULT_COMMAND_TOKEN_SIZE。
        """
        data = {**_VALID_COMMAND_DATA, "token_size": None}
        msg = KnowledgeImportCommandMessage(**data)
        assert msg.token_size == DEFAULT_COMMAND_TOKEN_SIZE

    def test_explicit_chunk_size_is_preserved(self):
        """测试目的：显式传入 chunk_size 时不应被默认值覆盖。
        预期结果：msg.chunk_size == 200。
        """
        data = {**_VALID_COMMAND_DATA, "chunk_size": 200}
        msg = KnowledgeImportCommandMessage(**data)
        assert msg.chunk_size == 200

    def test_default_chunk_strategy_is_character(self):
        """测试目的：未指定 chunk_strategy 时应默认为 CHARACTER。
        预期结果：msg.chunk_strategy == ChunkStrategyType.CHARACTER。
        """
        msg = KnowledgeImportCommandMessage(**_VALID_COMMAND_DATA)
        assert msg.chunk_strategy == ChunkStrategyType.CHARACTER

    def test_empty_task_uuid_raises_validation_error(self):
        """测试目的：task_uuid 为空字符串时应校验失败。
        预期结果：抛出 ValidationError。
        """
        data = {**_VALID_COMMAND_DATA, "task_uuid": ""}
        with pytest.raises(ValidationError):
            KnowledgeImportCommandMessage(**data)

    def test_negative_version_raises_validation_error(self):
        """测试目的：version < 1 时应校验失败。
        预期结果：抛出 ValidationError。
        """
        data = {**_VALID_COMMAND_DATA, "version": 0}
        with pytest.raises(ValidationError):
            KnowledgeImportCommandMessage(**data)

    def test_zero_document_id_raises_validation_error(self):
        """测试目的：document_id <= 0 时应校验失败。
        预期结果：抛出 ValidationError。
        """
        data = {**_VALID_COMMAND_DATA, "document_id": 0}
        with pytest.raises(ValidationError):
            KnowledgeImportCommandMessage(**data)


# ---- KnowledgeImportResultMessage 测试 -------------------------------------


class TestImportResultMessage:
    """KnowledgeImportResultMessage 测试。"""

    def test_build_sets_message_type(self):
        """测试目的：通过 build 创建的实例 message_type 应为 knowledge_import_result。
        预期结果：msg.message_type == "knowledge_import_result"。
        """
        msg = KnowledgeImportResultMessage.build(**_VALID_RESULT_KWARGS)
        assert msg.message_type == "knowledge_import_result"

    def test_build_auto_calculates_duration_ms(self):
        """测试目的：同时传入 started_at 和 occurred_at 时应自动计算 duration_ms。
        预期结果：duration_ms == 5000（5 秒）。
        """
        started = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        occurred = datetime(2025, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
        msg = KnowledgeImportResultMessage.build(
            **_VALID_RESULT_KWARGS, started_at=started, occurred_at=occurred,
        )
        assert msg.duration_ms == 5000

    def test_build_duration_zero_when_no_started_at(self):
        """测试目的：未传 started_at 时 duration_ms 应为 0。
        预期结果：duration_ms == 0。
        """
        msg = KnowledgeImportResultMessage.build(**_VALID_RESULT_KWARGS)
        assert msg.duration_ms == 0

    def test_build_duration_non_negative_when_started_after_occurred(self):
        """测试目的：started_at 晚于 occurred_at 时 duration_ms 不应为负数。
        预期结果：duration_ms == 0。
        """
        occurred = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        started = datetime(2025, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
        msg = KnowledgeImportResultMessage.build(
            **_VALID_RESULT_KWARGS, started_at=started, occurred_at=occurred,
        )
        assert msg.duration_ms == 0

    def test_build_preserves_optional_fields(self):
        """测试目的：build 传入 filename、chunk_count 等可选字段时应保留。
        预期结果：msg.filename、chunk_count、vector_count 与传入值一致。
        """
        msg = KnowledgeImportResultMessage.build(
            **_VALID_RESULT_KWARGS,
            filename="test.pdf",
            chunk_count=10,
            vector_count=10,
            embedding_dim=768,
        )
        assert msg.filename == "test.pdf"
        assert msg.chunk_count == 10
        assert msg.vector_count == 10
        assert msg.embedding_dim == 768

    def test_build_defaults_optional_fields(self):
        """测试目的：不传可选字段时应使用默认值。
        预期结果：filename 为 None，chunk_count/vector_count/embedding_dim 为 0。
        """
        msg = KnowledgeImportResultMessage.build(**_VALID_RESULT_KWARGS)
        assert msg.filename is None
        assert msg.chunk_count == 0
        assert msg.vector_count == 0
        assert msg.embedding_dim == 0

    def test_occurred_at_defaults_to_utc(self):
        """测试目的：未传 occurred_at 时应自动使用当前 UTC 时间。
        预期结果：occurred_at 时区为 UTC 且与当前时间相差不超过 2 秒。
        """
        msg = KnowledgeImportResultMessage.build(**_VALID_RESULT_KWARGS)
        diff = abs((datetime.now(timezone.utc) - msg.occurred_at).total_seconds())
        assert diff < 2
