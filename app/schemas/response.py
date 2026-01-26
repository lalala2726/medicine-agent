"""
通用 API 响应模型

提供统一的 API 响应格式，包含状态码、消息、数据和时间戳。
"""

from datetime import datetime
from typing import Generic, TypeVar, Optional, Any

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

    @classmethod
    def success(cls, data: T = None, message: str = ResponseCode.SUCCESS.message) -> "ApiResponse[T]":
        """
        返回成功响应

        Args:
            data: 响应数据，任意类型
            message: 成功消息，默认使用 ResponseCode.SUCCESS.message

        Returns:
            ApiResponse: 状态码为 200 的成功响应对象

        Example:
            >>> ApiResponse.success(data={"id": 1}, message="查询成功")
            ApiResponse(code=200, message='查询成功', data={'id': 1}, timestamp=1737868800)
        """
        return cls(code=ResponseCode.SUCCESS, message=message, data=data)

    @classmethod
    def error(cls, message: str = "error", code: int = 500, data: T = None) -> "ApiResponse[T]":
        """
        返回错误响应

        Args:
            message: 错误消息
            code: 错误码，默认 500
            data: 附加数据，可选

        Returns:
            ApiResponse: 指定错误码的错误响应对象

        Example:
            >>> ApiResponse.error(message="用户不存在", code=404)
            ApiResponse(code=404, message='用户不存在', data=None, timestamp=1737868800)
        """
        return cls(code=code, message=message, data=data)
