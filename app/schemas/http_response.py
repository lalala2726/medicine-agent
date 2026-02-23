from __future__ import annotations

from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict

from app.core.codes import ResponseCode
from app.exception.exceptions import ServiceException


class HttpResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: int
    message: str
    data: Any = None
    timestamp: Optional[Any] = None

    @classmethod
    def from_response(cls, response: httpx.Response) -> "HttpResponse":
        """从 httpx 响应构建 HttpResponse。"""
        try:
            payload = response.json()
        except ValueError as exc:
            raise ServiceException(
                code=ResponseCode.OPERATION_FAILED,
                message=f"响应不是合法 JSON，status={response.status_code}",
            ) from exc
        return cls.model_validate(payload)

    def data_or_raise(self) -> Any:
        """当 code 非 200 时抛出业务异常，否则返回 data。"""
        if self.code != ResponseCode.SUCCESS.code:
            raise ServiceException(code=self.code, message=self.message)
        return self.data

    @classmethod
    def parse_data(cls, response: httpx.Response) -> Any:
        """解析响应并直接返回 data（非 200 会抛异常）。"""
        return cls.from_response(response).data_or_raise()
