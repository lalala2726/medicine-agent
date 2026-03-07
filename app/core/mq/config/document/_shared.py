"""文档链路 MQ 配置共享定义。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.mq.config.common import DEFAULT_PREFETCH_COUNT


@dataclass(frozen=True)
class DocumentRabbitMQSettingsBase:
    """文档链路 MQ 拓扑配置基类。"""

    exchange: str
    command_queue: str
    command_routing_key: str
    result_routing_key: str
    prefetch_count: int = DEFAULT_PREFETCH_COUNT
