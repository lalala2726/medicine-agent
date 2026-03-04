from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import util as importlib_util

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException

_DEFAULT_EXCHANGE = "knowledge.import"
_DEFAULT_QUEUE = "knowledge.import.submit.q"
_DEFAULT_ROUTING_KEY = "knowledge.import.submit"
_DEFAULT_PREFETCH_COUNT = 1
_DEFAULT_RETRY_DELAYS = (
    15, 15, 30, 180, 600, 1200, 1800, 1800, 1800, 3600, 10800, 10800, 10800, 21600, 21600
)
_DEFAULT_CALLBACK_TIMEOUT_SECONDS = 5


def _parse_bool(value: str | None, *, default: bool) -> bool:
    """
    功能描述:
        解析布尔配置值，支持常见的 true/false 文本形式。

    参数说明:
        value (str | None): 原始环境变量值。
        default (bool): 当 value 为空时返回的默认值。

    返回值:
        bool: 解析后的布尔结果。

    异常说明:
        无。无法识别时回退为 False。
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int(value: str | None, *, default: int, name: str) -> int:
    """
    功能描述:
        解析正整数配置项，支持为空时回退默认值。

    参数说明:
        value (str | None): 原始环境变量值。
        default (int): 默认值。
        name (str): 配置项名称，用于错误提示。

    返回值:
        int: 解析后的正整数值。

    异常说明:
        ServiceException: 配置值不是正整数时抛出。
    """
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{name} 必须是正整数",
        ) from exc
    if parsed <= 0:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{name} 必须大于 0",
        )
    return parsed


def _parse_duration_seconds(value: str, *, name: str) -> int:
    """
    功能描述:
        将带时间单位的字符串解析为秒数，支持 `s`、`m`、`h` 后缀。

    参数说明:
        value (str): 时间字符串，例如 `15s`、`3m`、`6h` 或 `30`（默认按秒）。
        name (str): 配置项名称，用于错误提示。

    返回值:
        int: 解析后的秒数（正整数）。

    异常说明:
        ServiceException: 值格式非法或秒数非正整数时抛出。
    """
    normalized = value.strip().lower()
    if not normalized:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{name} 存在空值",
        )

    multiplier = 1
    numeric_part = normalized
    if normalized.endswith("s"):
        numeric_part = normalized[:-1]
        multiplier = 1
    elif normalized.endswith("m"):
        numeric_part = normalized[:-1]
        multiplier = 60
    elif normalized.endswith("h"):
        numeric_part = normalized[:-1]
        multiplier = 3600

    try:
        number = int(numeric_part)
    except ValueError as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{name} 存在非法时间值: {value}",
        ) from exc

    if number <= 0:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{name} 必须为正整数时间",
        )
    return number * multiplier


def _parse_retry_delays(value: str | None) -> tuple[int, ...]:
    """
    功能描述:
        解析重试延迟配置，支持逗号分隔秒数列表。

    参数说明:
        value (str | None): 原始环境变量值，示例：`5,30,120`。

    返回值:
        tuple[int, ...]: 延迟秒数元组。

    异常说明:
        ServiceException: 存在非正整数项时抛出。
    """
    if value is None or value.strip() == "":
        return _DEFAULT_RETRY_DELAYS
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        return _DEFAULT_RETRY_DELAYS
    parsed_items = [
        _parse_duration_seconds(item, name="MQ_RETRY_DELAYS_SECONDS")
        for item in parts
    ]
    return tuple(parsed_items)


def has_rabbitmq_url_configured() -> bool:
    """
    功能描述:
        判断当前环境是否配置了 RabbitMQ 连接地址。

    参数说明:
        无。

    返回值:
        bool: 已配置返回 True，否则返回 False。

    异常说明:
        无。
    """
    return bool((os.getenv("RABBITMQ_URL") or "").strip())


def is_mq_consumer_enabled() -> bool:
    """
    功能描述:
        判断应用内 MQ 消费者是否启用。

    参数说明:
        无。读取环境变量 `MQ_CONSUMER_ENABLED`。

    返回值:
        bool: 启用返回 True，禁用返回 False。

    异常说明:
        无。
    """
    return _parse_bool(os.getenv("MQ_CONSUMER_ENABLED"), default=True)


def is_aio_pika_installed() -> bool:
    """
    功能描述:
        判断当前运行环境是否安装了 aio-pika 依赖。

    参数说明:
        无。

    返回值:
        bool: 已安装返回 True，否则返回 False。

    异常说明:
        无。
    """
    return importlib_util.find_spec("aio_pika") is not None


@dataclass(frozen=True)
class RabbitMQSettings:
    """
    功能描述:
        封装 RabbitMQ 发布与消费所需配置。

    参数说明:
        url (str): RabbitMQ 连接地址。
        exchange (str): 交换机名称。
        queue (str): 队列名称。
        routing_key (str): 路由键。
        prefetch_count (int): 消费预取条数。
        max_retries (int): 单消息最大重试次数。
        retry_delays_seconds (tuple[int, ...]): 重试间隔（秒）列表。
        callback_url (str): 导入结果回调地址。
        callback_timeout_seconds (int): 回调 HTTP 超时时间（秒）。
        callback_max_retries (int): 回调最大重试次数。
        callback_retry_delays_seconds (tuple[int, ...]): 回调重试间隔（秒）列表。

    返回值:
        无。该类用于承载配置数据。

    异常说明:
        无。
    """

    url: str
    exchange: str
    queue: str
    routing_key: str
    prefetch_count: int
    max_retries: int
    retry_delays_seconds: tuple[int, ...]
    callback_url: str
    callback_timeout_seconds: int
    callback_max_retries: int
    callback_retry_delays_seconds: tuple[int, ...]


@lru_cache(maxsize=1)
def get_rabbitmq_settings() -> RabbitMQSettings:
    """
    功能描述:
        加载并缓存 RabbitMQ 配置，供发布器与消费者统一读取。

    参数说明:
        无。配置来源于环境变量。

    返回值:
        RabbitMQSettings: 解析后的配置对象。

    异常说明:
        ServiceException:
            - 未配置 `RABBITMQ_URL` 时抛出；
            - 未配置 `KNOWLEDGE_IMPORT_CALLBACK_URL` 时抛出；
            - 整数配置非法时抛出。
    """
    url = (os.getenv("RABBITMQ_URL") or "").strip()
    if not url:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="未配置 RABBITMQ_URL，无法提交异步导入任务",
        )
    exchange = (os.getenv("RABBITMQ_EXCHANGE") or _DEFAULT_EXCHANGE).strip()
    queue = (os.getenv("RABBITMQ_QUEUE") or _DEFAULT_QUEUE).strip()
    routing_key = (os.getenv("RABBITMQ_ROUTING_KEY") or _DEFAULT_ROUTING_KEY).strip()
    prefetch_count = _parse_positive_int(
        os.getenv("RABBITMQ_PREFETCH_COUNT"),
        default=_DEFAULT_PREFETCH_COUNT,
        name="RABBITMQ_PREFETCH_COUNT",
    )
    retry_delays_seconds = _parse_retry_delays(os.getenv("MQ_RETRY_DELAYS_SECONDS"))
    max_retries = _parse_positive_int(
        os.getenv("MQ_MAX_RETRIES"),
        default=len(retry_delays_seconds),
        name="MQ_MAX_RETRIES",
    )
    callback_url = (os.getenv("KNOWLEDGE_IMPORT_CALLBACK_URL") or "").strip()
    if not callback_url:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message="未配置 KNOWLEDGE_IMPORT_CALLBACK_URL，无法回调导入结果",
        )
    callback_timeout_seconds = _parse_positive_int(
        os.getenv("KNOWLEDGE_IMPORT_CALLBACK_TIMEOUT_SECONDS"),
        default=_DEFAULT_CALLBACK_TIMEOUT_SECONDS,
        name="KNOWLEDGE_IMPORT_CALLBACK_TIMEOUT_SECONDS",
    )
    callback_retry_delays_seconds = _parse_retry_delays(
        os.getenv("KNOWLEDGE_CALLBACK_RETRY_DELAYS_SECONDS")
        or os.getenv("MQ_RETRY_DELAYS_SECONDS")
    )
    callback_max_retries = _parse_positive_int(
        os.getenv("KNOWLEDGE_CALLBACK_MAX_RETRIES"),
        default=len(callback_retry_delays_seconds),
        name="KNOWLEDGE_CALLBACK_MAX_RETRIES",
    )
    return RabbitMQSettings(
        url=url,
        exchange=exchange,
        queue=queue,
        routing_key=routing_key,
        prefetch_count=prefetch_count,
        max_retries=max_retries,
        retry_delays_seconds=retry_delays_seconds,
        callback_url=callback_url,
        callback_timeout_seconds=callback_timeout_seconds,
        callback_max_retries=callback_max_retries,
        callback_retry_delays_seconds=callback_retry_delays_seconds,
    )
