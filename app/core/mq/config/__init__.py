"""MQ 配置子包。"""

from app.core.mq.config.settings import (
    RabbitMQSettings,
    get_rabbitmq_settings,
    has_rabbitmq_url_configured,
    is_aio_pika_installed,
    is_mq_consumer_enabled,
)
from app.core.mq.config.topology import (
    MQ_EXCHANGE_KNOWLEDGE_IMPORT,
    MQ_QUEUE_IMPORT_COMMAND,
    MQ_ROUTING_KEY_IMPORT_COMMAND,
    MQ_ROUTING_KEY_IMPORT_RESULT,
)

__all__ = [
    "RabbitMQSettings",
    "get_rabbitmq_settings",
    "has_rabbitmq_url_configured",
    "is_aio_pika_installed",
    "is_mq_consumer_enabled",
    "MQ_EXCHANGE_KNOWLEDGE_IMPORT",
    "MQ_QUEUE_IMPORT_COMMAND",
    "MQ_ROUTING_KEY_IMPORT_COMMAND",
    "MQ_ROUTING_KEY_IMPORT_RESULT",
]
