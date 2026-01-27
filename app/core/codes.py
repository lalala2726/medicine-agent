from enum import IntEnum


class ResponseCode(IntEnum):
    SUCCESS = (200, "操作成功")
    BAD_REQUEST = (400, "请求错误")
    OPERATION_FAILED = (400, "操作失败")
    UNAUTHORIZED = (401, "未认证或登录已失效")
    FORBIDDEN = (403, "无权限访问")
    NOT_FOUND = (404, "资源不存在")
    ERROR = (500, "服务器内部异常")
    INTERNAL_ERROR = (500, "服务器内部异常")

    def __new__(cls, code: int, message: str):
        obj = int.__new__(cls, code)
        obj._value_ = code
        obj.message = message
        return obj

    @property
    def code(self) -> int:
        return int(self)
