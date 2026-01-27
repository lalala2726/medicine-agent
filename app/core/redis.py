import os
from functools import lru_cache
from typing import Optional

from redis import Redis

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException


def _parse_int(value: Optional[str], name: str, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ServiceException(
            message=f"{name} must be an integer",
            code=ResponseCode.INTERNAL_ERROR,
        ) from exc


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_redis_connection() -> Redis:
    """Create and cache a Redis connection for RQ usage."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return Redis.from_url(redis_url)

    host = os.getenv("REDIS_HOST", "localhost")
    port = _parse_int(os.getenv("REDIS_PORT"), "REDIS_PORT", 6379)
    db = _parse_int(os.getenv("REDIS_DB"), "REDIS_DB", 0)
    password = os.getenv("REDIS_PASSWORD")
    ssl_enabled = _parse_bool(os.getenv("REDIS_SSL"))

    # decode_responses defaults to False to keep payloads as bytes for RQ.
    return Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        ssl=ssl_enabled,
    )
