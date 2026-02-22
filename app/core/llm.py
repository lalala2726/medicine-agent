import os
from typing import Any, Optional

from langchain.agents import create_agent as langchain_create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import SecretStr

from app.schemas.memory import Memory

# Load from environment variables
DEFAULT_CHAT_MODEL = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-max")
DEFAULT_IMAGE_MODEL = os.getenv("DASHSCOPE_IMAGE_MODEL", "qwen3-vl-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
DEFAULT_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def _normalize_message_signature(message: Any) -> tuple[str, str]:
    """提取消息类型与文本内容，用于判断记忆前缀是否已存在。"""

    raw_type = getattr(message, "type", None)
    message_type = str(raw_type or message.__class__.__name__).lower()
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return message_type, content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict):
                text_parts.append(str(item.get("text") or ""))
        return message_type, "".join(text_parts)
    return message_type, str(content)


def _has_memory_prefix(messages: list[Any], memory_messages: list[Any]) -> bool:
    """判断当前状态消息是否已以前缀方式包含 memory。"""

    if len(messages) < len(memory_messages):
        return False
    for index, memory_message in enumerate(memory_messages):
        if _normalize_message_signature(messages[index]) != _normalize_message_signature(memory_message):
            return False
    return True


def _build_memory_inject_middleware(store: Memory):
    """构建 before_model 中间件：把 memory 作为消息前缀注入。"""

    @before_model
    def _inject_memory(state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        _ = runtime
        memory_messages = list(store.messages or [])
        if not memory_messages:
            return None
        state_messages = list(state.get("messages") or [])
        if not state_messages:
            return None
        if _has_memory_prefix(state_messages, memory_messages):
            return None
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *memory_messages,
                *state_messages,
            ]
        }

    return _inject_memory


def create_agent_instance(
        *,
        llm: Any,
        store: Memory | None = None,
        tools: list[Any] | tuple[Any, ...] | None = None,
        system_prompt: Any = None,
        middleware: list[Any] | tuple[Any, ...] = (),
        response_format: Any = None,
        state_schema: type[Any] | None = None,
        context_schema: type[Any] | None = None,
        checkpointer: Any = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        debug: bool = False,
        name: str | None = None,
        cache: Any = None,
        **kwargs: Any,
) -> Any:
    """
    创建 Agent 实例（支持业务 Memory 预注入）。

    Args:
        llm: 预创建好的聊天模型实例（或模型名）。
        store: 业务记忆对象；仅用于注入 state.messages 前缀，不透传官方 `store`。
        tools: 可选工具列表。
        system_prompt: 可选系统提示。
        middleware: 可选中间件列表。
        response_format: 可选结构化响应配置。
        state_schema: 可选状态 schema。
        context_schema: 可选上下文 schema。
        checkpointer: 可选短期记忆 checkpointer。
        interrupt_before: 可选中断节点（before）。
        interrupt_after: 可选中断节点（after）。
        debug: 是否开启调试。
        name: 可选 agent 名称。
        cache: 可选缓存对象。
        **kwargs: 其余透传 `langchain.agents.create_agent` 的参数。

    Returns:
        Any: 由 LangChain `create_agent` 返回的可执行 Agent 实例。
    """

    resolved_middleware = list(middleware)
    if store is not None and store.messages:
        resolved_middleware.insert(0, _build_memory_inject_middleware(store))

    return langchain_create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=tuple(resolved_middleware),
        response_format=response_format,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,
        **kwargs,
    )


def _resolve_extra_body(
        model_kwargs: dict[str, Any],
        *,
        extra_body: Optional[dict[str, Any]],
        think: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    从 `model_kwargs` 中提取并合并 `extra_body`，同时按需开启深度思考。

    Args:
        model_kwargs: 当前 `model_kwargs` 字典（会复制后处理，不会原地修改入参）。
            兼容从该字典中读取旧写法 `extra_body`。
        extra_body: 调用方显式传入的扩展参数。
        think: 是否开启深度思考；开启时会强制写入 `enable_thinking=True`。

    Returns:
        tuple[dict[str, Any], dict[str, Any] | None]:
            - 第一个元素为清理后的 `model_kwargs`（已移除 `extra_body`）；
            - 第二个元素为合并后的 `extra_body`（无内容时为 None）。
    """

    cleaned_model_kwargs = dict(model_kwargs)
    nested_extra_body = dict(cleaned_model_kwargs.pop("extra_body", None) or {})
    if extra_body:
        nested_extra_body.update(extra_body)
    if think:
        nested_extra_body["enable_thinking"] = True
    return cleaned_model_kwargs, (nested_extra_body or None)


def create_chat_model(
        model: str = DEFAULT_CHAT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
        extra_body: Optional[dict[str, Any]] = None,
        think: bool = False,
        **kwargs
) -> ChatOpenAI:
    """
    创建聊天模型客户端

    Args:
        model: 模型名称，默认使用环境变量 DASHSCOPE_CHAT_MODEL
        api_key: API密钥，默认使用环境变量 DASHSCOPE_API_KEY
        base_url: API基础URL，默认使用环境变量 DASHSCOPE_BASE_URL
        response_format: 响应格式配置，如 {"type": "json_object"}
        extra_body: 透传给模型提供方的扩展参数，如 {"enable_thinking": True}
        think: 是否开启深度思考。为 True 时自动透传 `extra_body.enable_thinking=True`
        **kwargs: 其他传递给 ChatOpenAI 的参数

    Returns:
        ChatOpenAI: 聊天模型客户端

    Raises:
        RuntimeError: 当 API_KEY 未设置时
    """
    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    raw_model_kwargs = kwargs.pop("model_kwargs", None)
    model_kwargs = dict(raw_model_kwargs or {})
    if response_format:
        model_kwargs["response_format"] = response_format
    model_kwargs, resolved_extra_body = _resolve_extra_body(
        model_kwargs,
        extra_body=extra_body,
        think=think,
    )
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    if resolved_extra_body:
        # 使用显式参数传递 `extra_body`，避免 langsmith 对 model_kwargs 的告警。
        kwargs["extra_body"] = resolved_extra_body
    # 默认开启 stream_usage，优先获取模型返回的真实 token 消耗。
    # 若供应商未返回 usage，业务层会使用 tiktoken 估算兜底。
    kwargs.setdefault("stream_usage", True)


    return ChatOpenAI(
        model=model,
        api_key=SecretStr(key),
        base_url=base_url or DEFAULT_BASE_URL,
        **kwargs,
    )


def create_chat_mode(*args, **kwargs) -> ChatOpenAI:
    """
    `create_chat_model` 的兼容别名。

    Args:
        *args: 透传给 `create_chat_model` 的位置参数。
        **kwargs: 透传给 `create_chat_model` 的关键字参数。

    Returns:
        ChatOpenAI: 聊天模型客户端实例。
    """

    return create_chat_model(*args, **kwargs)


def create_embedding_model(
        model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = 1024,
        **kwargs
) -> OpenAIEmbeddings:
    """
    创建向量嵌入客户端

    Args:
        model: 嵌入模型名称，默认使用环境变量 DASHSCOPE_EMBEDDING_MODEL
        api_key: API密钥，默认使用环境变量 DASHSCOPE_API_KEY
        base_url: API基础URL，默认使用环境变量 DASHSCOPE_BASE_URL
        dimensions: 向量维度，范围 128-4096 且为偶数
        **kwargs: 其他传递给 OpenAIEmbeddings 的参数

    Returns:
        OpenAIEmbeddings: 向量嵌入客户端

    Raises:
        RuntimeError: 当 API_KEY 未设置时
        ValueError: 当维度不在有效范围时
    """
    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    if dimensions is not None:
        if dimensions < 128 or dimensions > 4096 or dimensions % 2 != 0:
            raise ValueError("Dimensions must be between 128 and 4096 and a multiple of 2")

    return OpenAIEmbeddings(
        model=model,
        check_embedding_ctx_length=False,
        api_key=SecretStr(key),
        base_url=base_url or DEFAULT_BASE_URL,
        dimensions=dimensions,
        **kwargs,
    )


def create_image_model(
        model: str = DEFAULT_IMAGE_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
        **kwargs
) -> ChatOpenAI:
    """
    创建图像理解模型客户端

    Args:
        model: 模型名称，默认使用环境变量 DASHSCOPE_IMAGE_MODEL
        api_key: API密钥
        base_url: API基础URL
        response_format: 响应格式配置
        **kwargs: 其他参数

    Returns:
        ChatOpenAI: 图像模型客户端
    """
    return create_chat_model(
        model=model,
        api_key=api_key,
        base_url=base_url,
        response_format=response_format,
        **kwargs
    )
