import os
from functools import lru_cache
from typing import Any, Callable, Optional

from rq import Queue
from rq.job import Job

from app.core.redis import get_redis_connection


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@lru_cache(maxsize=8)
def get_rq_queue(name: Optional[str] = None) -> Queue:
    """Return a cached RQ queue bound to the shared Redis connection."""
    queue_name = name or os.getenv("RQ_QUEUE_NAME", "default")
    default_timeout = _parse_int(os.getenv("RQ_DEFAULT_TIMEOUT"), 180)
    return Queue(
        name=queue_name,
        connection=get_redis_connection(),
        default_timeout=default_timeout,
    )


def enqueue_job(
    func: Callable[..., Any],
    *args: Any,
    queue_name: Optional[str] = None,
    **kwargs: Any,
) -> Job:
    """Enqueue a job to RQ using the shared queue.

    Example:
        enqueue_job(my_task, 1, 2, job_timeout=300)
    """
    queue = get_rq_queue(queue_name)
    return queue.enqueue(func, *args, **kwargs)
