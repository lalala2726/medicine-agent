from __future__ import annotations

import asyncio

from loguru import logger

from app.core.mq.config.settings import (
    has_rabbitmq_url_configured,
    is_aio_pika_installed,
    is_mq_consumer_enabled,
)
from app.core.mq.consumers.import_consumer import run_import_consumer

_consumer_task: asyncio.Task[None] | None = None


async def start_import_consumer_if_enabled() -> None:
    """
    功能描述:
        按配置条件启动应用内导入消费者任务，避免重复启动。

    参数说明:
        无。启用开关与连接配置由环境变量控制。

    返回值:
        None: 启动流程结束无返回值。

    异常说明:
        无。未满足条件时仅记录日志并跳过。
    """
    global _consumer_task
    if not is_mq_consumer_enabled():
        logger.info("MQ_CONSUMER_ENABLED=false，跳过导入消费者启动")
        return
    if not has_rabbitmq_url_configured():
        logger.warning("未配置 RABBITMQ_URL，跳过导入消费者启动")
        return
    if not is_aio_pika_installed():
        logger.warning("未安装 aio-pika，跳过导入消费者启动")
        return
    if _consumer_task and not _consumer_task.done():
        logger.info("导入 MQ 消费者已在运行，跳过重复启动")
        return
    _consumer_task = asyncio.create_task(
        run_import_consumer(),
        name="knowledge-import-command-consumer",
    )
    logger.info("知识库导入命令 MQ 消费者已启动")


async def stop_import_consumer() -> None:
    """
    功能描述:
        停止应用内导入消费者任务并等待其优雅退出。

    参数说明:
        无。

    返回值:
        None: 停止流程结束无返回值。

    异常说明:
        无。取消异常会被吞掉以保证 shutdown 流程继续执行。
    """
    global _consumer_task
    if _consumer_task is None:
        return
    if not _consumer_task.done():
        _consumer_task.cancel()
        try:
            await _consumer_task
        except asyncio.CancelledError:
            pass
    _consumer_task = None
    logger.info("知识库导入命令 MQ 消费者已停止")
