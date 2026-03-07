from __future__ import annotations

import asyncio

from loguru import logger

from app.core.mq.config.common import (
    has_rabbitmq_url_configured,
    is_aio_pika_installed,
    is_chunk_add_consumer_enabled,
    is_chunk_rebuild_consumer_enabled,
    is_import_consumer_enabled,
)
from app.core.mq.consumers.document.chunk_add_consumer import run_chunk_add_consumer
from app.core.mq.consumers.document.chunk_rebuild_consumer import (
    run_chunk_rebuild_consumer,
)
from app.core.mq.consumers.document.import_consumer import run_import_consumer

_consumer_task: asyncio.Task[None] | None = None
_chunk_rebuild_consumer_task: asyncio.Task[None] | None = None
_chunk_add_consumer_task: asyncio.Task[None] | None = None


async def start_import_consumer_if_enabled() -> None:
    """按配置条件启动应用内导入消费者任务，避免重复启动。

    Returns:
        None: 启动流程结束无返回值。
    """
    global _consumer_task
    if not is_import_consumer_enabled():
        logger.info("导入消费者开关已关闭，跳过导入消费者启动")
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
    """停止应用内导入消费者任务并等待其优雅退出。

    Returns:
        None: 停止流程结束无返回值。
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


async def start_chunk_rebuild_consumer_if_enabled() -> None:
    """按配置条件启动应用内切片重建消费者任务，避免重复启动。

    Returns:
        None: 启动流程结束无返回值。
    """
    global _chunk_rebuild_consumer_task
    if not is_chunk_rebuild_consumer_enabled():
        logger.info("切片重建消费者开关已关闭，跳过切片重建消费者启动")
        return
    if not has_rabbitmq_url_configured():
        logger.warning("未配置 RABBITMQ_URL，跳过切片重建消费者启动")
        return
    if not is_aio_pika_installed():
        logger.warning("未安装 aio-pika，跳过切片重建消费者启动")
        return
    if _chunk_rebuild_consumer_task and not _chunk_rebuild_consumer_task.done():
        logger.info("切片重建 MQ 消费者已在运行，跳过重复启动")
        return
    _chunk_rebuild_consumer_task = asyncio.create_task(
        run_chunk_rebuild_consumer(),
        name="knowledge-chunk-rebuild-command-consumer",
    )
    logger.info("切片重建命令 MQ 消费者已启动")


async def stop_chunk_rebuild_consumer() -> None:
    """停止应用内切片重建消费者任务并等待其优雅退出。

    Returns:
        None: 停止流程结束无返回值。
    """
    global _chunk_rebuild_consumer_task
    if _chunk_rebuild_consumer_task is None:
        return
    if not _chunk_rebuild_consumer_task.done():
        _chunk_rebuild_consumer_task.cancel()
        try:
            await _chunk_rebuild_consumer_task
        except asyncio.CancelledError:
            pass
    _chunk_rebuild_consumer_task = None
    logger.info("切片重建命令 MQ 消费者已停止")


async def start_chunk_add_consumer_if_enabled() -> None:
    """按配置条件启动应用内手工新增切片消费者任务，避免重复启动。

    Returns:
        None: 启动流程结束无返回值。
    """
    global _chunk_add_consumer_task
    if not is_chunk_add_consumer_enabled():
        logger.info("手工新增切片消费者开关已关闭，跳过手工新增切片消费者启动")
        return
    if not has_rabbitmq_url_configured():
        logger.warning("未配置 RABBITMQ_URL，跳过手工新增切片消费者启动")
        return
    if not is_aio_pika_installed():
        logger.warning("未安装 aio-pika，跳过手工新增切片消费者启动")
        return
    if _chunk_add_consumer_task and not _chunk_add_consumer_task.done():
        logger.info("手工新增切片 MQ 消费者已在运行，跳过重复启动")
        return
    _chunk_add_consumer_task = asyncio.create_task(
        run_chunk_add_consumer(),
        name="knowledge-chunk-add-command-consumer",
    )
    logger.info("手工新增切片命令 MQ 消费者已启动")


async def stop_chunk_add_consumer() -> None:
    """停止应用内手工新增切片消费者任务并等待其优雅退出。

    Returns:
        None: 停止流程结束无返回值。
    """
    global _chunk_add_consumer_task
    if _chunk_add_consumer_task is None:
        return
    if not _chunk_add_consumer_task.done():
        _chunk_add_consumer_task.cancel()
        try:
            await _chunk_add_consumer_task
        except asyncio.CancelledError:
            pass
    _chunk_add_consumer_task = None
    logger.info("手工新增切片命令 MQ 消费者已停止")
