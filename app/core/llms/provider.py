from __future__ import annotations

from enum import Enum


class LlmProvider(str, Enum):
    """
    功能描述:
        定义聊天模型提供商枚举，统一约束工厂函数 `provider` 入参取值，
        避免调用方传入拼写错误的厂商标识。

    参数说明:
        无（枚举类型定义）。

    返回值:
        无（类型定义）。

    异常说明:
        无（异常由枚举解析阶段在调用方触发）。
    """

    OPENAI = "openai"
    ALIYUN = "aliyun"


def normalize_provider(provider: LlmProvider | str) -> LlmProvider:
    """
    功能描述:
        归一化厂商参数，支持 `LlmProvider` 或字符串输入，
        统一转换为 `LlmProvider` 枚举实例。

    参数说明:
        provider (LlmProvider | str): 厂商标识；
            - 当为 `LlmProvider` 时直接返回；
            - 当为字符串时按枚举值解析（大小写敏感，使用小写值）。

    返回值:
        LlmProvider: 归一化后的厂商枚举值。

    异常说明:
        ValueError: 当字符串不在支持范围内时抛出。
    """

    if isinstance(provider, LlmProvider):
        return provider
    return LlmProvider(provider)
