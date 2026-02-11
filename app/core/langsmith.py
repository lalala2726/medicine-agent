import os
from typing import Any

try:
    from langsmith import traceable as _traceable
except Exception:  # pragma: no cover - optional dependency fallback
    _traceable = None


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_langsmith_enabled() -> bool:
    """
    LangSmith tracing 开关：
    - LANGSMITH_TRACING=true（新）
    - LANGCHAIN_TRACING_V2=true（旧，兼容）
    """
    return _is_truthy(os.getenv("LANGSMITH_TRACING")) or _is_truthy(os.getenv("LANGCHAIN_TRACING_V2"))


def build_langsmith_runnable_config(
        run_name: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not is_langsmith_enabled():
        return None

    config: dict[str, Any] = {"run_name": run_name}
    if tags:
        config["tags"] = tags
    if metadata:
        config["metadata"] = metadata
    return config


def traceable(*args, **kwargs):
    """
    统一 traceable 装饰器入口：
    - 安装了 langsmith 则使用官方 traceable
    - 未安装时退化为 no-op，避免运行失败
    """
    if _traceable is None:
        def _decorator(func):
            return func

        return _decorator
    return _traceable(*args, **kwargs)
