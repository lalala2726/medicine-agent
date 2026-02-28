from __future__ import annotations

import os

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException


def resolve_required_env(name: str) -> str:
    """读取必填环境变量；为空时抛出配置异常。"""

    value = (os.getenv(name) or "").strip()
    if value:
        return value
    raise ServiceException(
        code=ResponseCode.INTERNAL_ERROR,
        message=f"{name} is not set",
    )


__all__ = ["resolve_required_env"]
