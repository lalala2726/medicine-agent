import os
from typing import Any

DEFAULT_CORS_ALLOW_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def load_cors_config() -> dict[str, Any]:
    allow_origins = _parse_csv_env("CORS_ALLOW_ORIGINS", [])
    allow_origin_regex = os.getenv(
        "CORS_ALLOW_ORIGIN_REGEX",
        DEFAULT_CORS_ALLOW_ORIGIN_REGEX,
    )
    if allow_origins:
        allow_origin_regex = None

    return {
        "allow_origins": allow_origins,
        "allow_origin_regex": allow_origin_regex,
        "allow_methods": _parse_csv_env("CORS_ALLOW_METHODS", ["*"]),
        "allow_headers": _parse_csv_env("CORS_ALLOW_HEADERS", ["*"]),
        "allow_credentials": _parse_bool_env("CORS_ALLOW_CREDENTIALS", True),
    }
