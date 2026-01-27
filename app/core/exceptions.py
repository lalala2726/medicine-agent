from typing import Any, Optional, Union

from app.core.codes import ResponseCode


class ServiceException(Exception):
    """业务异常，供全局异常处理器统一返回结构化响应。"""

    def __init__(
        self,
        message: Optional[str] = None,
        code: Union[int, ResponseCode] = ResponseCode.OPERATION_FAILED,
        data: Optional[Any] = None,
    ) -> None:
        resolved_message = message
        if resolved_message is None and isinstance(code, ResponseCode):
            resolved_message = code.message
        if resolved_message is None:
            resolved_message = "error"
        super().__init__(resolved_message)
        self.message = resolved_message
        self.code = int(code)
        self.data = data
