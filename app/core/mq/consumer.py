from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pydantic import ValidationError

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.core.mq.models import KnowledgeImportMessage
from app.core.mq.settings import RabbitMQSettings, get_rabbitmq_settings
from app.services.knowledge_base_service import import_knowledge_service


def _load_aio_pika() -> tuple[Any, Any]:
    """
    功能描述:
        懒加载 aio-pika 依赖，避免在模块导入阶段产生硬依赖。

    参数说明:
        无。

    返回值:
        tuple[Any, Any]:
            - connect_robust 函数
            - ExchangeType 枚举

    异常说明:
        ServiceException: 未安装 aio-pika 依赖时抛出。
    """
    try:
        from aio_pika import ExchangeType, connect_robust
    except Exception as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="缺少 aio-pika 依赖，无法启动 MQ 消费者",
        ) from exc
    return connect_robust, ExchangeType


def process_import_message_once(message: KnowledgeImportMessage) -> bool:
    """
    功能描述:
        执行单条导入消息的业务处理，调用现有导入服务完成下载、解析与切片打印。

    参数说明:
        message (KnowledgeImportMessage): 导入任务消息。

    返回值:
        bool: 处理成功返回 True，处理失败返回 False。

    异常说明:
        无。函数内部不抛出业务异常，统一通过返回值表示结果。
    """
    result = import_knowledge_service(
        knowledge_name=message.knowledge_name,
        document_id=message.document_id,
        file_url=[message.file_url],
        chunk_strategy=message.chunk_strategy,
        chunk_size=message.chunk_size,
        token_size=message.token_size,
    )
    failed_urls = result.get("failed_urls") or []
    return len(failed_urls) == 0


async def process_import_message_with_retry(
        message: KnowledgeImportMessage,
        *,
        max_retries: int,
        retry_delays_seconds: tuple[int, ...],
        process_func: Callable[[KnowledgeImportMessage], bool] = process_import_message_once,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> bool:
    """
    功能描述:
        对单条导入消息执行带重试的处理逻辑，失败后按配置间隔重试。

    参数说明:
        message (KnowledgeImportMessage): 导入任务消息。
        max_retries (int): 最大重试次数（不含首次执行）。
        retry_delays_seconds (tuple[int, ...]): 重试等待秒数列表。
        process_func (Callable[[KnowledgeImportMessage], bool]): 实际处理函数，默认值为 process_import_message_once。
        sleep_func (Callable[[float], Awaitable[None]]): 异步休眠函数，默认值为 asyncio.sleep。

    返回值:
        bool: 在允许次数内成功返回 True，全部尝试失败返回 False。

    异常说明:
        无。异常会被捕获并记录日志，不向上抛出。
    """
    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
        try:
            success = process_func(message)
            if success:
                return True
        except Exception as exc:
            logger.exception(
                "导入任务执行异常：task_uuid={}, document_id={}, file_url={}, attempt={}, error={}",
                message.task_uuid,
                message.document_id,
                message.file_url,
                attempt + 1,
                exc,
            )
        if attempt >= total_attempts - 1:
            break
        delay_index = min(attempt, max(0, len(retry_delays_seconds) - 1))
        await sleep_func(float(retry_delays_seconds[delay_index]))
    return False


def parse_import_message(body: bytes) -> KnowledgeImportMessage:
    """
    功能描述:
        将 RabbitMQ 原始消息体解析为导入任务模型。

    参数说明:
        body (bytes): MQ 消息体字节串。

    返回值:
        KnowledgeImportMessage: 解析后的导入任务消息对象。

    异常说明:
        ValidationError: JSON 结构非法或字段校验失败时抛出。
        JSONDecodeError: 消息体不是合法 JSON 时抛出。
    """
    payload = json.loads(body.decode("utf-8"))
    return KnowledgeImportMessage.model_validate(payload)


async def _handle_incoming_message(
        incoming: Any,
        settings: RabbitMQSettings,
) -> None:
    """
    功能描述:
        处理一条 RabbitMQ 消息并执行 ACK，确保失败消息不会无限堆积。

    参数说明:
        incoming (IncomingMessage): RabbitMQ 入站消息对象。
        settings (RabbitMQSettings): RabbitMQ 运行配置。

    返回值:
        None: 处理完成无返回值。

    异常说明:
        无。异常会在函数内部记录并最终 ACK。
    """
    try:
        message = parse_import_message(incoming.body)
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.error("导入消息格式非法，已丢弃并 ACK：error={}", exc)
        await incoming.ack()
        return

    success = await process_import_message_with_retry(
        message,
        max_retries=settings.max_retries,
        retry_delays_seconds=settings.retry_delays_seconds,
    )
    if not success:
        logger.error(
            "导入任务失败且重试耗尽，已 ACK：task_uuid={}, document_id={}, file_url={}",
            message.task_uuid,
            message.document_id,
            message.file_url,
        )
    await incoming.ack()


async def _consume_once(settings: RabbitMQSettings) -> None:
    """
    功能描述:
        建立 RabbitMQ 连接并持续消费导入队列消息，直到连接中断或任务取消。

    参数说明:
        settings (RabbitMQSettings): RabbitMQ 运行配置。

    返回值:
        None: 消费循环结束无返回值。

    异常说明:
        Exception: 连接失败或消费失败时抛出，由上层循环处理重连。
    """
    connect_robust, exchange_type_enum = _load_aio_pika()
    connection = await connect_robust(settings.url)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=settings.prefetch_count)
        exchange = await channel.declare_exchange(
            settings.exchange,
            exchange_type_enum.DIRECT,
            durable=True,
        )
        queue = await channel.declare_queue(settings.queue, durable=True)
        await queue.bind(exchange, routing_key=settings.routing_key)
        async with queue.iterator() as queue_iter:
            async for incoming in queue_iter:
                await _handle_incoming_message(incoming, settings)


async def run_import_consumer() -> None:
    """
    功能描述:
        启动导入消息消费者并保持常驻，消费异常时自动等待后重连。

    参数说明:
        无。配置从环境变量加载。

    返回值:
        None: 常驻任务正常退出时无返回值。

    异常说明:
        无。非取消异常会被捕获并重试。
    """
    settings = get_rabbitmq_settings()
    while True:
        try:
            await _consume_once(settings)
        except asyncio.CancelledError:
            logger.info("知识库导入 MQ 消费者收到取消信号，准备停止")
            raise
        except Exception as exc:
            logger.exception("知识库导入 MQ 消费异常，5 秒后重连：error={}", exc)
            await asyncio.sleep(5)
