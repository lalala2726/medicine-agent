"""MQ 公共配置常量与运行时开关。"""

from __future__ import annotations

import os
from importlib import util as importlib_util


def has_rabbitmq_url_configured() -> bool:
    """检查是否已配置 RabbitMQ 地址。"""
    return bool((os.getenv("RABBITMQ_URL") or "").strip())


def is_import_consumer_enabled() -> bool:
    """检查是否启用应用内导入 MQ 消费者。"""
    return IMPORT_CONSUMER_ENABLED


def is_chunk_rebuild_consumer_enabled() -> bool:
    """检查是否启用应用内切片重建 MQ 消费者。"""
    return CHUNK_REBUILD_CONSUMER_ENABLED


def is_chunk_add_consumer_enabled() -> bool:
    """检查是否启用应用内手工新增切片 MQ 消费者。"""
    return CHUNK_ADD_CONSUMER_ENABLED


def is_aio_pika_installed() -> bool:
    """检查是否已安装 aio-pika 依赖。"""
    return importlib_util.find_spec("aio_pika") is not None


# 文档链路消费者预取数量。
DEFAULT_PREFETCH_COUNT = 1

# 应用内文档导入 MQ 消费者是否启动。
IMPORT_CONSUMER_ENABLED = True

# 应用内切片重建 MQ 消费者是否启动。
CHUNK_REBUILD_CONSUMER_ENABLED = True

# 应用内手工新增切片 MQ 消费者是否启动。
CHUNK_ADD_CONSUMER_ENABLED = True
