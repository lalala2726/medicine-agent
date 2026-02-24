import os
from typing import Any

DEFAULT_CORS_ALLOW_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    """
    解析逗号分隔的环境变量为字符串列表。

    Args:
        name: 环境变量名称。
        default: 环境变量不存在或解析为空时使用的默认值。

    Returns:
        list[str]: 解析后的非空字符串列表。
    """
    value = os.getenv(name)
    if value is None:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


def _parse_bool_env(name: str, default: bool) -> bool:
    """
    解析布尔类型环境变量。

    支持：
    - 真值：`1/true/yes/on`
    - 假值：`0/false/no/off`
    其他非法值回落到默认值。

    Args:
        name: 环境变量名称。
        default: 变量不存在或值非法时使用的默认值。

    Returns:
        bool: 解析后的布尔值。
    """
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
    """
    加载 FastAPI CORS 中间件配置。

    规则：
    1. 若 `CORS_ALLOW_ORIGINS` 显式配置且非空，则优先使用 origins 列表；
    2. 否则使用 `CORS_ALLOW_ORIGIN_REGEX`（默认仅允许本地开发域名）；
    3. methods/headers/credentials 由对应环境变量解析，未配置使用默认值。

    Returns:
        dict[str, Any]: 可直接传给 `CORSMiddleware` 的配置字典。
    """
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
