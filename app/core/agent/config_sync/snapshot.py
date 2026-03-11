from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.core.database import get_redis_connection

#: Redis 中保存 Agent 全量运行时配置的固定 key。
AGENT_CONFIG_REDIS_KEY = "agent:config:all"


class AgentChatModelSlot(str, Enum):
    """管理助手聊天模型槽位。"""

    ROUTE = "routeModel"
    CHAT = "chatModel"
    BUSINESS_SIMPLE = "businessNodeSimpleModel"
    BUSINESS_COMPLEX = "businessNodeComplexModel"


class AgentImageModelSlot(str, Enum):
    """图片识别模型槽位。"""

    RECOGNITION = "imageRecognitionModel"


class AgentEmbeddingModelSlot(str, Enum):
    """向量模型槽位。"""

    EMBEDDING = "embeddingModel"


class AgentConfigSource(str, Enum):
    """当前内存快照来源。"""

    REDIS = "redis"
    LOCAL_FALLBACK = "local_fallback"


class AgentConfigLoadReason(str, Enum):
    """Agent 配置加载失败原因。"""

    REDIS_KEY_MISSING = "redis_key_missing"
    INVALID_UTF8 = "invalid_utf8"
    UNSUPPORTED_PAYLOAD_TYPE = "unsupported_payload_type"
    REDIS_READ_FAILED = "redis_read_failed"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"


#: Agent 配置加载失败原因到中文日志文案的映射。
_LOAD_REASON_LABELS: dict[AgentConfigLoadReason, str] = {
    AgentConfigLoadReason.REDIS_KEY_MISSING: "Redis Key 不存在",
    AgentConfigLoadReason.INVALID_UTF8: "配置内容不是合法 UTF-8",
    AgentConfigLoadReason.UNSUPPORTED_PAYLOAD_TYPE: "配置内容类型不支持",
    AgentConfigLoadReason.REDIS_READ_FAILED: "读取 Redis 失败",
    AgentConfigLoadReason.INVALID_JSON: "配置内容不是合法 JSON",
    AgentConfigLoadReason.INVALID_SCHEMA: "配置结构校验失败",
}

#: 管理端 provider 名称到内部 provider 标识的归一化映射。
_PROVIDER_ALIAS_MAP: dict[str, str] = {
    "openai": "openai",
    "aliyun": "aliyun",
    "qwen": "aliyun",
    "dashscope": "aliyun",
    "volcengine": "volcengine",
    "ark": "volcengine",
}


def _strip_optional_str(value: Any) -> str | None:
    """将可选字符串值规整为去空白后的字符串。

    Args:
        value: 原始输入值，允许为 ``None``、字符串或其他可转字符串对象。

    Returns:
        去除首尾空白后的字符串；若输入为空值或空白字符串则返回 ``None``。
    """

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


class AgentModelRuntimeConfig(BaseModel):
    """Redis 中单个运行时模型配置。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    provider: str | None = None
    model: str | None = None
    model_type: str | None = Field(default=None, alias="modelType")
    base_url: str | None = Field(default=None, alias="baseUrl")
    api_key: str | None = Field(default=None, alias="apiKey")
    support_reasoning: bool | None = Field(default=None, alias="supportReasoning")
    support_vision: bool | None = Field(default=None, alias="supportVision")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any) -> str | None:
        """归一化 Redis 里的 provider 名称。"""

        normalized = _strip_optional_str(value)
        if normalized is None:
            return None
        alias = _PROVIDER_ALIAS_MAP.get(normalized.lower())
        if alias is None:
            raise ValueError(f"Unsupported agent config provider: {normalized}")
        return alias

    @field_validator("model", "model_type", "base_url", "api_key", mode="before")
    @classmethod
    def _normalize_optional_str(cls, value: Any) -> str | None:
        """归一化可选字符串字段。"""

        return _strip_optional_str(value)

    @model_validator(mode="after")
    def _validate_runtime_shape(self) -> AgentModelRuntimeConfig:
        """校验运行时模型配置的最小完整性。"""

        has_any_runtime_value = any(
            value is not None
            for value in (self.provider, self.model, self.base_url, self.api_key)
        )
        if has_any_runtime_value and (self.provider is None or self.model is None):
            raise ValueError("Agent runtime config requires both provider and model")
        return self


class AgentModelSlotConfig(BaseModel):
    """业务槽位配置。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    reasoning_enabled: bool | None = Field(default=None, alias="reasoningEnabled")
    max_tokens: int | None = Field(default=None, alias="maxTokens")
    temperature: float | None = None
    model: AgentModelRuntimeConfig | None = None


class KnowledgeBaseAgentConfig(BaseModel):
    """知识库配置。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    embedding_dim: int | None = Field(default=None, alias="embeddingDim")
    embedding_model: AgentModelSlotConfig | None = Field(default=None, alias="embeddingModel")
    rerank_model: AgentModelSlotConfig | None = Field(default=None, alias="rerankModel")


class AdminAssistantAgentConfig(BaseModel):
    """管理助手配置。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    route_model: AgentModelSlotConfig | None = Field(default=None, alias="routeModel")
    business_node_simple_model: AgentModelSlotConfig | None = Field(
        default=None,
        alias="businessNodeSimpleModel",
    )
    business_node_complex_model: AgentModelSlotConfig | None = Field(
        default=None,
        alias="businessNodeComplexModel",
    )
    chat_model: AgentModelSlotConfig | None = Field(default=None, alias="chatModel")


class ImageRecognitionAgentConfig(BaseModel):
    """图片识别配置。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    image_recognition_model: AgentModelSlotConfig | None = Field(
        default=None,
        alias="imageRecognitionModel",
    )


class ChatHistorySummaryAgentConfig(BaseModel):
    """聊天历史总结配置。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    chat_history_summary_model: AgentModelSlotConfig | None = Field(
        default=None,
        alias="chatHistorySummaryModel",
    )


class ChatTitleAgentConfig(BaseModel):
    """聊天标题生成配置。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    chat_title_model: AgentModelSlotConfig | None = Field(
        default=None,
        alias="chatTitleModel",
    )


class AgentConfigSnapshot(BaseModel):
    """Agent 运行时配置快照。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    updated_at: datetime | None = Field(default=None, alias="updatedAt")
    updated_by: str | None = Field(default=None, alias="updatedBy")
    knowledge_base: KnowledgeBaseAgentConfig | None = Field(default=None, alias="knowledgeBase")
    admin_assistant: AdminAssistantAgentConfig | None = Field(default=None, alias="adminAssistant")
    image_recognition: ImageRecognitionAgentConfig | None = Field(
        default=None,
        alias="imageRecognition",
    )
    chat_history_summary: ChatHistorySummaryAgentConfig | None = Field(
        default=None,
        alias="chatHistorySummary",
    )
    chat_title: ChatTitleAgentConfig | None = Field(
        default=None,
        alias="chatTitle",
    )

    @field_validator("updated_by", mode="before")
    @classmethod
    def _normalize_updated_by(cls, value: Any) -> str | None:
        """归一化更新人字段。"""

        return _strip_optional_str(value)

    def get_chat_slot(self, slot: AgentChatModelSlot) -> AgentModelSlotConfig | None:
        """根据聊天槽位枚举读取管理助手对应槽位配置。

        Args:
            slot: 目标聊天槽位。

        Returns:
            命中的槽位配置；若当前快照没有管理助手配置或该槽位为空则返回 ``None``。
        """

        admin_assistant = self.admin_assistant
        if admin_assistant is None:
            return None
        if slot is AgentChatModelSlot.ROUTE:
            return admin_assistant.route_model
        if slot is AgentChatModelSlot.CHAT:
            return admin_assistant.chat_model
        if slot is AgentChatModelSlot.BUSINESS_SIMPLE:
            return admin_assistant.business_node_simple_model
        if slot is AgentChatModelSlot.BUSINESS_COMPLEX:
            return admin_assistant.business_node_complex_model
        return None

    def get_image_slot(self) -> AgentModelSlotConfig | None:
        """读取图片识别槽位配置。

        Returns:
            图片识别模型槽位配置；未配置时返回 ``None``。
        """

        image_recognition = self.image_recognition
        if image_recognition is None:
            return None
        return image_recognition.image_recognition_model

    def get_embedding_slot(self) -> AgentModelSlotConfig | None:
        """读取知识库 embedding 槽位配置。

        Returns:
            知识库 embedding 模型槽位配置；未配置时返回 ``None``。
        """

        knowledge_base = self.knowledge_base
        if knowledge_base is None:
            return None
        return knowledge_base.embedding_model

    def get_summary_slot(self) -> AgentModelSlotConfig | None:
        """读取聊天历史总结槽位配置。

        Returns:
            聊天历史总结模型槽位配置；未配置时返回 ``None``。
        """

        chat_history_summary = self.chat_history_summary
        if chat_history_summary is None:
            return None
        return chat_history_summary.chat_history_summary_model

    def get_title_slot(self) -> AgentModelSlotConfig | None:
        """读取聊天标题生成槽位配置。

        Returns:
            聊天标题生成模型槽位配置；未配置时返回 ``None``。
        """

        chat_title = self.chat_title
        if chat_title is None:
            return None
        return chat_title.chat_title_model


class AgentConfigLoadError(RuntimeError):
    """Redis 配置加载失败。"""

    def __init__(self, reason: AgentConfigLoadReason, message: str) -> None:
        """初始化加载异常。

        Args:
            reason: 失败原因枚举，供日志记录与测试断言使用。
            message: 脱敏后的异常消息，不包含 Redis 原始配置内容。
        """

        super().__init__(message)
        self.reason = reason


#: 保护进程内 Agent 配置快照读写的一把锁。
_CONFIG_LOCK = threading.RLock()
#: 当前生效的进程内配置快照。
_current_snapshot: AgentConfigSnapshot | None = None
#: 当前生效快照来源，便于日志和调试定位。
_current_source = AgentConfigSource.LOCAL_FALLBACK


def _build_local_fallback_snapshot() -> AgentConfigSnapshot:
    """构造本地 `.env` 兜底场景使用的空快照。

    Returns:
        一个不包含 Redis 业务配置的本地兜底快照。
    """

    return AgentConfigSnapshot(
        updated_at=datetime.now(timezone.utc),
        updated_by="local_env_fallback",
    )


def _get_load_reason_label(reason: AgentConfigLoadReason) -> str:
    """返回加载失败原因对应的中文日志文案。

    Args:
        reason: 加载失败原因枚举。

    Returns:
        适合写入日志的中文原因描述。
    """

    return _LOAD_REASON_LABELS.get(reason, "未知错误")


def _decode_redis_payload(*, raw_payload: Any, redis_key: str) -> str:
    """将 Redis 原始返回值解码为 JSON 文本。

    Args:
        raw_payload: Redis ``GET`` 返回的原始值。
        redis_key: 当前读取的 Redis key，用于错误提示。

    Returns:
        解码后的 JSON 字符串。

    Raises:
        AgentConfigLoadError: 当 key 不存在、编码非法或 payload 类型不支持时抛出。
    """

    if raw_payload is None:
        raise AgentConfigLoadError(
            AgentConfigLoadReason.REDIS_KEY_MISSING,
            f"Redis key {redis_key} is missing",
        )
    if isinstance(raw_payload, bytes):
        try:
            return raw_payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AgentConfigLoadError(
                AgentConfigLoadReason.INVALID_UTF8,
                "Agent config payload is not valid utf-8",
            ) from exc
    if isinstance(raw_payload, str):
        return raw_payload
    raise AgentConfigLoadError(
        AgentConfigLoadReason.UNSUPPORTED_PAYLOAD_TYPE,
        f"Unsupported agent config payload type: {type(raw_payload)!r}",
    )


def _load_snapshot_from_redis(*, redis_key: str) -> AgentConfigSnapshot:
    """从 Redis 读取并反序列化 Agent 配置快照。

    Args:
        redis_key: 要读取的 Redis key。

    Returns:
        通过 schema 校验后的 Agent 配置快照。

    Raises:
        AgentConfigLoadError: 当 Redis 读取、JSON 解析或 schema 校验失败时抛出。
    """

    try:
        raw_payload = get_redis_connection().get(redis_key)
    except Exception as exc:
        raise AgentConfigLoadError(
            AgentConfigLoadReason.REDIS_READ_FAILED,
            f"Failed to read redis key {redis_key}",
        ) from exc

    payload = _decode_redis_payload(raw_payload=raw_payload, redis_key=redis_key)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AgentConfigLoadError(
            AgentConfigLoadReason.INVALID_JSON,
            "Agent config payload is not valid JSON",
        ) from exc
    try:
        snapshot = AgentConfigSnapshot.model_validate(data)
    except ValidationError as exc:
        raise AgentConfigLoadError(
            AgentConfigLoadReason.INVALID_SCHEMA,
            "Agent config payload schema is invalid",
        ) from exc
    return snapshot


def _set_current_snapshot(snapshot: AgentConfigSnapshot, *, source: AgentConfigSource) -> None:
    """原子更新进程内当前快照状态。

    Args:
        snapshot: 要写入的新快照。
        source: 该快照的来源。
    """

    global _current_snapshot, _current_source
    _current_snapshot = snapshot
    _current_source = source


def initialize_agent_config_snapshot() -> AgentConfigSnapshot:
    """初始化进程内 Agent 配置快照。

    启动时优先尝试从 Redis 读取；若 Redis 不可用、key 缺失或数据非法，
    则回退到本地 `.env` 兜底快照。

    Returns:
        当前初始化后生效的 Agent 配置快照副本。
    """

    with _CONFIG_LOCK:
        if _current_snapshot is not None:
            return _current_snapshot.model_copy(deep=True)

    try:
        snapshot = _load_snapshot_from_redis(redis_key=AGENT_CONFIG_REDIS_KEY)
    except AgentConfigLoadError as exc:
        snapshot = _build_local_fallback_snapshot()
        with _CONFIG_LOCK:
            _set_current_snapshot(snapshot, source=AgentConfigSource.LOCAL_FALLBACK)
        logger.warning(
            "Agent 配置初始化完成：来源={}，redis_key={}，错误原因={}",
            AgentConfigSource.LOCAL_FALLBACK.value,
            AGENT_CONFIG_REDIS_KEY,
            _get_load_reason_label(exc.reason),
        )
        return snapshot.model_copy(deep=True)

    with _CONFIG_LOCK:
        _set_current_snapshot(snapshot, source=AgentConfigSource.REDIS)
    logger.info(
        "Agent 配置初始化完成：来源={}，redis_key={}",
        AgentConfigSource.REDIS.value,
        AGENT_CONFIG_REDIS_KEY,
    )
    return snapshot.model_copy(deep=True)


def get_current_agent_config_snapshot() -> AgentConfigSnapshot:
    """读取当前生效的 Agent 配置快照。

    Returns:
        当前生效快照的深拷贝副本。
    """

    with _CONFIG_LOCK:
        if _current_snapshot is not None:
            return _current_snapshot.model_copy(deep=True)
    return initialize_agent_config_snapshot()


def refresh_agent_config_snapshot(*, redis_key: str) -> bool:
    """在收到 MQ 刷新通知后重新拉取 Redis 配置并更新本地快照。

    Args:
        redis_key: 需要重新读取的 Redis key。

    Returns:
        当成功从 Redis 读取并替换当前快照时返回 ``True``；否则返回 ``False``。
    """

    try:
        snapshot = _load_snapshot_from_redis(redis_key=redis_key)
    except AgentConfigLoadError as exc:
        logger.warning(
            "Agent 配置刷新失败，继续保留当前快照：redis_key={}，错误原因={}",
            redis_key,
            _get_load_reason_label(exc.reason),
        )
        return False

    with _CONFIG_LOCK:
        _set_current_snapshot(snapshot, source=AgentConfigSource.REDIS)
    logger.info(
        "Agent 配置刷新已生效：来源={}，redis_key={}",
        AgentConfigSource.REDIS.value,
        redis_key,
    )
    return True


def clear_agent_config_snapshot_state() -> None:
    """清理进程内 Agent 配置快照状态。

    该函数主要供测试使用，用于隔离用例之间的模块级全局状态。
    """

    global _current_snapshot, _current_source
    with _CONFIG_LOCK:
        _current_snapshot = None
        _current_source = AgentConfigSource.LOCAL_FALLBACK
