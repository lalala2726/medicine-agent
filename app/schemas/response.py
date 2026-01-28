"""
通用 API 响应模型

提供统一的 API 响应格式，包含状态码、消息、数据和时间戳。
"""

from datetime import datetime
from typing import Generic, TypeVar, Optional, Any, overload

from pydantic import BaseModel, Field

from app.core.codes import ResponseCode

# 泛型类型变量，用于支持任意类型的响应数据
T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """
    通用 API 响应模型

    Attributes:
        code: 状态码，默认 200 表示成功
        message: 响应消息，用于描述操作结果
        data: 响应数据，支持任意类型
        timestamp: Unix 时间戳（秒级）

    Generic:
        T: 响应数据的类型，通过泛型支持类型提示
    """

    code: int = Field(default=200, description="状态码")
    message: str = Field(default="success", description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp()), description="时间戳")

    def model_dump(self, **kwargs):  # type: ignore[override]
        kwargs["exclude_none"] = True
        return super().model_dump(**kwargs)

    @classmethod
    def success(cls, data: T = None, message: str = ResponseCode.SUCCESS.message) -> "ApiResponse[T]":
        return cls(code=ResponseCode.SUCCESS, message=message, data=data)

    @classmethod
    @overload
    def error(cls, response: ResponseCode, data: T = None) -> "ApiResponse[T]":
        ...

    @classmethod
    @overload
    def error(cls, response: ResponseCode, message: str, data: T = None) -> "ApiResponse[T]":
        ...

    @classmethod
    def error(
            cls,
            response: ResponseCode,
            message: Optional[str] = None,
            data: T = None,
    ) -> "ApiResponse[T]":
        resolved_message = message if message is not None else response.message
        return cls(code=response.code, message=resolved_message, data=data)


class Response(ApiResponse[Any]):
    """API response alias for global handlers."""
