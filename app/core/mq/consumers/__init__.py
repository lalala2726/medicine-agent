"""MQ 消费者子包。"""

from app.core.mq.consumers.chunk_add_consumer import (
    parse_chunk_add_command,
    run_chunk_add_consumer,
)
from app.core.mq.consumers.chunk_rebuild_consumer import (
    parse_chunk_rebuild_command,
    run_chunk_rebuild_consumer,
)
from app.core.mq.consumers.import_consumer import parse_import_command, run_import_consumer
from app.core.mq.consumers.lifecycle import (
    start_chunk_add_consumer_if_enabled,
    start_chunk_rebuild_consumer_if_enabled,
    start_import_consumer_if_enabled,
    stop_chunk_add_consumer,
    stop_chunk_rebuild_consumer,
    stop_import_consumer,
)

__all__ = [
    "parse_chunk_add_command",
    "run_chunk_add_consumer",
    "parse_chunk_rebuild_command",
    "run_chunk_rebuild_consumer",
    "parse_import_command",
    "run_import_consumer",
    "start_chunk_add_consumer_if_enabled",
    "start_chunk_rebuild_consumer_if_enabled",
    "start_import_consumer_if_enabled",
    "stop_chunk_add_consumer",
    "stop_chunk_rebuild_consumer",
    "stop_import_consumer",
]
