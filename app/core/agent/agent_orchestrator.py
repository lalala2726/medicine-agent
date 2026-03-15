"""
管理助手流式输出编排层（SSE transport/orchestration）。

这个文件只负责“如何把事件流式推给前端”，包括：
1. workflow 事件消费（messages / values / emitted / error / done）；
2. 事件到 SSE 协议的封包与输出；
3. 结束收尾、尾事件 drain、fallback 文本输出。

这个文件不负责“业务节点如何执行”与“工具如何调用”，例如：
- LLM invoke/tool 调用循环；
- tool 参数编排与工具执行失败重试；
- 具体业务状态字段如何产生。

放置建议：
- 需要复用 SSE 输出流程、事件分发、收尾逻辑的代码放这里；
- 需要做模型调用、工具调用、业务执行策略的代码不要放这里。
"""

import asyncio
import inspect
import json
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Awaitable, Callable

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from loguru import logger

from app.core.agent.agent_event_bus import (
    drain_final_sse_responses,
    reset_final_response_queue,
    reset_status_emitter,
    set_final_response_queue,
    set_status_emitter,
)
from app.schemas.sse_response import Action, AssistantResponse, Content, MessageType
from app.utils.streaming_utils import extract_text

StreamEvent = tuple[str, Any]
GraphEventPayload = tuple[str, Any]
InitialEmittedEvent = AssistantResponse | dict[str, Any]
OnAnswerCompletedCallback = Callable[..., None | Awaitable[None]]

EVENT_EMITTED = "emitted"
EVENT_GRAPH = "graph"
EVENT_ERROR = "error"
EVENT_DONE = "done"

GRAPH_MODE_MESSAGES = "messages"
GRAPH_MODE_VALUES = "values"

UNKNOWN_MODEL_NAME = "unknown"


@dataclass(frozen=True)
class AssistantStreamConfig:
    """
    助手流式输出的可配置参数。

    这是对外暴露的配置对象，路由层只需要提供业务相关回调，
    通用流式引擎会根据这些回调完成事件消费、SSE 封包和收尾处理。

    Attributes:
        workflow: 具体的 workflow 对象（通常是 LangGraph 编译后的 graph）。
        build_initial_state: 根据用户问题构造初始状态的函数。
        extract_final_content: 当没有 token 流输出时，从最终状态提取兜底文本。
        should_stream_token: 判定某个 graph 节点 token 是否应输出给前端。
        build_stream_config: 生成 astream 调用配置（可返回 None 表示不传 config）。
        invoke_sync: 无 astream 能力时的同步执行入口（通常内部调用 graph.invoke）。
        map_exception: 将异常映射为前端可读错误文案的函数。
        initial_emitted_events: 流开始前先注入的事件（用于会话创建成功等前置通知）。
        hide_node_types: 对哪些事件类型隐藏 `node` 字段，默认隐藏 function_call。
        stream_modes: astream 订阅模式，默认 messages + values。
        response_headers: StreamingResponse 的响应头，默认包含禁缓存和禁代理缓冲。
    """

    # 流式执行主体，决定事件从哪里产出。
    workflow: Any
    # 构造 workflow 初始状态：输入是问题字符串，输出是状态字典。
    build_initial_state: Callable[[str], dict[str, Any]]
    # 当没有 token 输出时，从最终状态提取用户可见文本。
    extract_final_content: Callable[[dict[str, Any]], str]
    # token 级过滤器：控制哪些节点 token 可以流向前端。
    should_stream_token: Callable[[str | None, dict[str, Any]], bool]
    # 生成 astream 的 config 参数；为 None 或返回 None 时不传。
    build_stream_config: Callable[[], dict | None] | None
    # 回退执行器：无 astream 时通过该函数同步执行 workflow。
    invoke_sync: Callable[[dict[str, Any]], dict[str, Any]]
    # 异常映射器：把内部异常转成统一错误文案。
    map_exception: Callable[[Exception], str]
    # 可选收尾回调：在流结束时回调完整 answer/thinking 文本、执行追踪与 token 汇总。
    # 回调兼容 2/3/4/5 参：2参(answer,trace)、3参(answer,trace,has_error)、
    # 4参(answer,trace,token_usage,has_error)、
    # 5参(answer,trace,token_usage,has_error,thinking_text)。
    on_answer_completed: OnAnswerCompletedCallback | None = None
    # 流开始前注入的事件列表（例如会话创建成功事件），会按给定顺序输出。
    initial_emitted_events: tuple[InitialEmittedEvent, ...] = field(
        default_factory=tuple
    )
    # 这些事件类型会隐藏 content.node，避免暴露内部节点标识。
    hide_node_types: set[MessageType] = field(
        default_factory=lambda: {MessageType.FUNCTION_CALL}
    )
    # astream 的 stream_mode 配置，决定消费哪些事件通道。
    stream_modes: tuple[str, ...] = ("messages", "values")
    # SSE 响应头配置，默认关闭缓存并关闭反向代理缓冲。
    response_headers: dict[str, str] = field(
        default_factory=lambda: {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@dataclass
class StreamRuntimeState:
    """
    流式会话运行时状态。

    Attributes:
        latest_state: 最近一次收到的 graph state（通常来自 values 事件）。
        has_streamed_output: 是否已经输出过 token 级 answer。
        has_emitted_error: 是否已经输出过错误 answer。
    """

    latest_state: dict[str, Any]
    has_streamed_output: bool = False
    has_emitted_error: bool = False
    aggregated_answer_parts: list[str] = field(default_factory=list)
    aggregated_answer_text: str = ""
    aggregated_thinking_parts: list[str] = field(default_factory=list)
    aggregated_thinking_text: str = ""
    active_tool_calls: int = 0


@dataclass
class EventProcessResult:
    """
    单次事件处理结果。

    Attributes:
        rendered_events: 当前事件产生的 SSE 文本列表。
        should_break: 主循环是否应在当前事件后进入 done 收尾流程。
    """

    rendered_events: list[str] = field(default_factory=list)
    should_break: bool = False


def serialize_sse(payload: AssistantResponse) -> str:
    """
    将 AssistantResponse 序列化为 SSE 行文本。

    说明：
    - 采用 `exclude_none=True`，避免输出无意义空字段，减少前端解析噪音。
    - 统一使用 `data: <json>\\n\\n` 格式，符合标准 SSE 协议。
    """

    return (
        f"data: {json.dumps(payload.model_dump(mode='json', exclude_none=True), ensure_ascii=False)}\n\n"
    )


def build_answer_sse(text: str, is_end: bool) -> str:
    """
    构造 answer 类型的 SSE 文本。

    Args:
        text: 本次要输出的文本片段。
        is_end: 是否为流式结束包。
    """

    payload = AssistantResponse(
        content=Content(text=text),
        type=MessageType.ANSWER,
        is_end=is_end,
    )
    return serialize_sse(payload)


def _resolve_message_type(raw_type: Any) -> MessageType:
    """
    将输入事件类型解析为 MessageType。

    非法值会回退为 `status`，保证通用流式层具备容错能力，
    不因上游事件类型不规范而中断主流程。
    """

    if isinstance(raw_type, MessageType):
        return raw_type

    normalized_type = str(raw_type or MessageType.STATUS.value)
    try:
        return MessageType(normalized_type)
    except ValueError:
        return MessageType.STATUS


def _resolve_timestamp(raw_timestamp: Any) -> int | None:
    """解析输入时间戳，非法值回退为 None（由模型默认值填充）。"""

    if isinstance(raw_timestamp, int):
        return raw_timestamp
    return None


def _resolve_meta(raw_meta: Any) -> dict[str, Any] | None:
    """解析输入的 meta 字段，仅接受字典类型。"""

    if isinstance(raw_meta, dict):
        return raw_meta
    return None


def _resolve_action(raw_action: Any) -> Action | None:
    """解析输入的 action 字段，仅接受合法动作对象。"""

    if isinstance(raw_action, Action):
        return raw_action
    if isinstance(raw_action, dict):
        try:
            return Action.model_validate(raw_action)
        except Exception:
            return None
    return None


def _to_non_negative_int(value: Any) -> int | None:
    """将值解析为非负整数。"""

    if value is None:
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    if resolved < 0:
        return None
    return resolved


def build_emitted_response(
        event_payload: dict[str, Any], hide_node_types: set[MessageType]
) -> AssistantResponse | None:
    """
    将状态发射器事件（兼容旧信封与 AssistantResponse 风格）转换为 AssistantResponse。

    Args:
        event_payload: 发射器推送的原始事件。
        hide_node_types: 需要隐藏 `node` 字段的事件类型集合。

    Returns:
        AssistantResponse | None: 可解析时返回标准响应模型；结构不合法时返回 None 并忽略。

    说明：
    - `function_call` 等事件按配置隐藏 `node`，避免把“工具节点内部标识”暴露给前端。
    - 如果事件结构不合法（例如 content 不是字典），选择忽略而非抛错，
      保障主流式链路稳定。
    - 自定义发射事件无论传入何种 `is_end`，都会在这里强制归一为 False，
      保证流结束包仅由收尾流程输出一次。
    """

    if not isinstance(event_payload, dict):
        return None

    content = event_payload.get("content")
    if not isinstance(content, dict):
        return None

    message_type = _resolve_message_type(event_payload.get("type"))

    node = content.get("node")
    if message_type in hide_node_types:
        node = None

    payload_kwargs: dict[str, Any] = {}
    resolved_timestamp = _resolve_timestamp(event_payload.get("timestamp"))
    if resolved_timestamp is not None:
        payload_kwargs["timestamp"] = resolved_timestamp

    # 优先读取新字段 `meta`，兼容读取旧字段 `extra`。
    raw_meta = event_payload.get("meta")
    if raw_meta is None:
        raw_meta = event_payload.get("extra")
    resolved_meta = _resolve_meta(raw_meta)
    if resolved_meta is not None:
        payload_kwargs["meta"] = resolved_meta
    resolved_action = _resolve_action(event_payload.get("action"))
    if resolved_action is not None:
        payload_kwargs["action"] = resolved_action

    return AssistantResponse(
        content=Content(
            text=content.get("text"),
            node=node,
            parent_node=content.get("parent_node"),
            state=content.get("state"),
            message=content.get("message"),
            result=content.get("result"),
            name=content.get("name"),
            arguments=content.get("arguments"),
        ),
        type=message_type,
        is_end=False,
        **payload_kwargs,
    )


def build_emitted_sse(
        event_payload: dict[str, Any], hide_node_types: set[MessageType]
) -> str | None:
    """将状态发射器事件转换为 SSE 文本。"""

    payload = build_emitted_response(event_payload, hide_node_types)
    if payload is None:
        return None
    return serialize_sse(payload)


def _normalize_initial_event_payload(event: InitialEmittedEvent) -> dict[str, Any] | None:
    """
    归一化预注入事件负载。

    支持两种输入：
    - AssistantResponse：按统一 JSON 结构导出；
    - dict：直接作为 emitted payload 使用。
    """

    if isinstance(event, AssistantResponse):
        return event.model_dump(mode="json", exclude_none=True)
    if isinstance(event, dict):
        return event
    return None


def _normalize_execution_trace_item(
        raw_item: Any,
        *,
        fallback_sequence: int,
) -> dict[str, Any] | None:
    """
    归一化单条 execution_trace 记录。

    Args:
        raw_item: 原始执行追踪项。
        fallback_sequence: 当上游未提供合法 `sequence` 时使用的兜底顺序值。

    Returns:
        dict[str, Any] | None:
            合法记录返回标准化字典；非法记录返回 None。
    """

    if not isinstance(raw_item, dict):
        return None

    node_name = str(raw_item.get("node_name") or "").strip()
    if not node_name:
        return None

    raw_sequence = _to_non_negative_int(raw_item.get("sequence"))
    sequence = raw_sequence if raw_sequence is not None and raw_sequence >= 1 else fallback_sequence
    model_name = str(raw_item.get("model_name") or UNKNOWN_MODEL_NAME).strip() or UNKNOWN_MODEL_NAME
    status = str(raw_item.get("status") or "success").strip().lower()
    normalized_status = status if status in {"success", "error"} else "success"
    tool_calls = raw_item.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
    raw_node_context = raw_item.get("node_context")
    node_context = raw_node_context if isinstance(raw_node_context, dict) else None

    return {
        "sequence": sequence,
        "node_name": node_name,
        "model_name": model_name,
        "status": normalized_status,
        "output_text": str(raw_item.get("output_text") or ""),
        "llm_usage_complete": bool(raw_item.get("llm_usage_complete", True)),
        "llm_token_usage": raw_item.get("llm_token_usage"),
        "tool_calls": tool_calls,
        "node_context": node_context,
    }


def _build_execution_trace_summary(latest_state: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    从最新状态提取 execution_trace 汇总。

    Args:
        latest_state: graph 最新状态。

    Returns:
        list[dict[str, Any]] | None: 归一化后的执行追踪列表，无有效项时返回 None。
    """

    raw_items = latest_state.get("execution_traces")
    if not isinstance(raw_items, list):
        return None

    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        normalized_item = _normalize_execution_trace_item(
            item,
            fallback_sequence=index,
        )
        if normalized_item is not None:
            normalized_items.append(normalized_item)
    return normalized_items or None


def _build_token_usage_summary(latest_state: dict[str, Any]) -> dict[str, Any] | None:
    """
    从最新状态提取 token_usage 汇总。

    Args:
        latest_state: graph 最新状态。

    Returns:
        dict[str, Any] | None: 标准化 token_usage；缺失或非法时返回 None。
    """

    raw_usage = latest_state.get("token_usage")
    if not isinstance(raw_usage, dict):
        return None

    prompt_tokens = _to_non_negative_int(raw_usage.get("prompt_tokens"))
    completion_tokens = _to_non_negative_int(raw_usage.get("completion_tokens"))
    total_tokens = _to_non_negative_int(raw_usage.get("total_tokens"))
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    resolved_prompt = prompt_tokens or 0
    resolved_completion = completion_tokens or 0
    resolved_total = total_tokens
    if resolved_total is None:
        resolved_total = resolved_prompt + resolved_completion

    raw_breakdown = raw_usage.get("node_breakdown")
    node_breakdown = raw_breakdown if isinstance(raw_breakdown, list) else []

    return {
        "prompt_tokens": resolved_prompt,
        "completion_tokens": resolved_completion,
        "total_tokens": resolved_total,
        "is_complete": bool(raw_usage.get("is_complete", True)),
        "node_breakdown": node_breakdown,
    }


def _append_answer_text(runtime_state: StreamRuntimeState, text: str) -> str:
    """
    向聚合答案缓冲区追加文本，并对“全量快照重复”做增量裁剪。

    说明：
    - 某些模型/链路会在 messages 事件中重复返回“截至当前的全量文本”，
      若直接拼接会出现同一段内容重复多次；
    - 当新文本以前缀方式包含了已聚合文本时，仅追加新增的 delta。

    Args:
        runtime_state: 流式运行时状态。
        text: 待追加的文本片段。

    Returns:
        str: 实际新增并可对外输出的文本；若无新增则返回空字符串。
    """

    raw_text = str(text or "")
    if not raw_text:
        return ""

    delta_text = raw_text
    existing_text = runtime_state.aggregated_answer_text
    if existing_text and raw_text.startswith(existing_text):
        delta_text = raw_text[len(existing_text):]

    if not delta_text:
        return ""

    runtime_state.aggregated_answer_parts.append(delta_text)
    runtime_state.aggregated_answer_text += delta_text
    return delta_text


def _append_thinking_text(runtime_state: StreamRuntimeState, text: str) -> str:
    """
    向聚合思考缓冲区追加文本，并对“全量快照重复”做增量裁剪。

    Args:
        runtime_state: 流式运行时状态。
        text: 待追加的思考文本片段。

    Returns:
        str: 实际新增的文本；若无新增则返回空字符串。
    """

    raw_text = str(text or "")
    if not raw_text:
        return ""

    delta_text = raw_text
    existing_text = runtime_state.aggregated_thinking_text
    if existing_text and raw_text.startswith(existing_text):
        delta_text = raw_text[len(existing_text):]

    if not delta_text:
        return ""

    runtime_state.aggregated_thinking_parts.append(delta_text)
    runtime_state.aggregated_thinking_text += delta_text
    return delta_text


def _update_tool_call_depth(runtime_state: StreamRuntimeState, payload: Any) -> None:
    """
    根据 emitted 的 function_call 事件维护工具调用深度计数。

    Args:
        runtime_state: 流式运行时状态。
        payload: emitted 事件原始负载。

    Returns:
        None
    """

    if not isinstance(payload, dict):
        return
    if str(payload.get("type") or "").strip() != MessageType.FUNCTION_CALL.value:
        return

    content = payload.get("content")
    if not isinstance(content, dict):
        return

    state = str(content.get("state") or "").strip()
    if state == "start":
        runtime_state.active_tool_calls += 1
        return
    if state == "end":
        runtime_state.active_tool_calls = max(0, runtime_state.active_tool_calls - 1)


def handle_graph_message_chunk(
        *,
        chunk: Any,
        runtime_state: StreamRuntimeState,
        should_stream_token: Callable[[str | None, dict[str, Any]], bool],
) -> EventProcessResult:
    """
    处理 graph `messages` 事件。

    Args:
        chunk: graph 透传的 `(message_chunk, metadata)` 二元组。
        runtime_state: 当前运行时状态（用于读取 latest_state 并更新输出标记）。
        should_stream_token: 业务侧 token 输出判定函数。

    Returns:
        EventProcessResult: 包含本次事件生成的 SSE 文本（可能为空）。
    """

    result = EventProcessResult()

    if not isinstance(chunk, tuple) or len(chunk) != 2:
        return result

    message_chunk, metadata = chunk
    stream_node: str | None = None
    if isinstance(metadata, dict):
        stream_node = metadata.get("langgraph_node")

    should_emit = should_stream_token(stream_node, runtime_state.latest_state)
    if not should_emit:
        return result

    # 工具执行期间会触发工具内部 LLM 调用，这些 token 不应直接对前端输出，
    # 否则会与最终 supervisor 汇总结果重复。
    if runtime_state.active_tool_calls > 0:
        return result

    token_text = extract_text(message_chunk)
    if not token_text:
        return result

    delta_text = _append_answer_text(runtime_state, token_text)
    if not delta_text:
        return result

    runtime_state.has_streamed_output = True
    result.rendered_events.append(build_answer_sse(delta_text, False))
    return result


def _process_graph_values_event(chunk: Any, runtime_state: StreamRuntimeState) -> None:
    """
    处理 graph `values` 事件。

    values 事件用于更新最新状态快照，不直接输出给前端。
    """

    if isinstance(chunk, dict):
        merged_state = dict(runtime_state.latest_state or {})
        merged_state.update(chunk)
        runtime_state.latest_state = merged_state


def _process_graph_event(
        payload: GraphEventPayload,
        runtime_state: StreamRuntimeState,
        should_stream_token: Callable[[str | None, dict[str, Any]], bool],
) -> EventProcessResult:
    """
    处理 graph 事件分支，并按 mode 分发到 message/values 处理器。
    """

    result = EventProcessResult()
    if not isinstance(payload, tuple) or len(payload) != 2:
        return result

    mode, chunk = payload
    if mode == GRAPH_MODE_MESSAGES:
        return handle_graph_message_chunk(
            chunk=chunk,
            runtime_state=runtime_state,
            should_stream_token=should_stream_token,
        )
    if mode == GRAPH_MODE_VALUES:
        _process_graph_values_event(chunk, runtime_state)
    return result


def _process_stream_event(
        *,
        event_type: str,
        payload: Any,
        runtime_state: StreamRuntimeState,
        should_stream_token: Callable[[str | None, dict[str, Any]], bool],
        hide_node_types: set[MessageType],
        map_exception: Callable[[Exception], str],
) -> EventProcessResult:
    """
    统一处理单个队列事件。

    该函数是主循环和 drain 流程共享的分发入口，
    用于避免两处重复写 `emitted/graph/error` 分支。
    """

    if event_type == EVENT_EMITTED:
        _update_tool_call_depth(runtime_state, payload)
        emitted_response = build_emitted_response(payload, hide_node_types)
        result = EventProcessResult()
        if emitted_response is None:
            return result

        # 自定义 answer 事件也视作“已输出内容”，避免 done 后 fallback 重复输出。
        if (
                emitted_response.type == MessageType.ANSWER
                and isinstance(emitted_response.content.text, str)
                and emitted_response.content.text
        ):
            delta_text = _append_answer_text(runtime_state, emitted_response.content.text)
            if delta_text:
                runtime_state.has_streamed_output = True
                emitted_response = emitted_response.model_copy(
                    update={
                        "content": emitted_response.content.model_copy(update={"text": delta_text}),
                    }
                )
            else:
                return result
        elif (
                emitted_response.type == MessageType.THINKING
                and isinstance(emitted_response.content.text, str)
                and emitted_response.content.text
        ):
            delta_thinking = _append_thinking_text(runtime_state, emitted_response.content.text)
            if delta_thinking:
                emitted_response = emitted_response.model_copy(
                    update={
                        "content": emitted_response.content.model_copy(update={"text": delta_thinking}),
                    }
                )
            else:
                return result

        result.rendered_events.append(serialize_sse(emitted_response))
        return result

    if event_type == EVENT_GRAPH:
        return _process_graph_event(payload, runtime_state, should_stream_token)

    if event_type == EVENT_ERROR:
        runtime_state.has_emitted_error = True
        result = EventProcessResult()
        message = map_exception(payload)
        delta_text = _append_answer_text(runtime_state, message)
        if delta_text:
            result.rendered_events.append(build_answer_sse(delta_text, False))
        return result

    if event_type == EVENT_DONE:
        return EventProcessResult(should_break=True)

    return EventProcessResult()


async def drain_pending_events(
        *,
        queue: asyncio.Queue[StreamEvent],
        runtime_state: StreamRuntimeState,
        should_stream_token: Callable[[str | None, dict[str, Any]], bool],
        hide_node_types: set[MessageType],
        map_exception: Callable[[Exception], str],
) -> EventProcessResult:
    """
    在 `done` 事件后，尽可能消费并输出队列里的尾部事件。

    说明：
    - 该函数只负责“清空尾部队列并生成可输出事件”，不决定主流程是否结束。
    - 运行时状态（latest_state、错误标记、token 标记）通过 `runtime_state` 原地更新。
    """

    result = EventProcessResult()

    while True:
        try:
            pending_type, pending_payload = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        pending_result = _process_stream_event(
            event_type=pending_type,
            payload=pending_payload,
            runtime_state=runtime_state,
            should_stream_token=should_stream_token,
            hide_node_types=hide_node_types,
            map_exception=map_exception,
        )
        for rendered_item in pending_result.rendered_events:
            result.rendered_events.append(rendered_item)

    return result


def _build_stream_kwargs(config: AssistantStreamConfig) -> dict[str, Any]:
    """
    组装 workflow `astream` 所需参数。

    stream_mode 固定来源于配置；
    config.build_stream_config 返回值为空时不注入，保持调用参数干净。
    """

    stream_kwargs: dict[str, Any] = {"stream_mode": list(config.stream_modes)}
    if config.build_stream_config is not None:
        stream_config = config.build_stream_config()
        if stream_config:
            stream_kwargs["config"] = stream_config
    return stream_kwargs


async def _produce_workflow_events(
        *,
        queue: asyncio.Queue[StreamEvent],
        state: dict[str, Any],
        runtime_state: StreamRuntimeState,
        config: AssistantStreamConfig,
) -> None:
    """
    生产 workflow 事件并写入队列。

    流程：
    1. 优先走 `workflow.astream`（messages + values）
    2. 若无 astream 则走同步 invoke 回退
    3. 任意异常统一写入 `error` 事件
    4. 最终一定写入 `done`，驱动消费方进入收尾流程
    """

    try:
        if hasattr(config.workflow, "astream"):
            stream_kwargs = _build_stream_kwargs(config)
            async for mode, chunk in config.workflow.astream(state, **stream_kwargs):
                await queue.put((EVENT_GRAPH, (mode, chunk)))
        else:
            runtime_state.latest_state = await run_in_threadpool(config.invoke_sync, state)
    except Exception as exc:
        logger.opt(exception=exc).error("Assistant workflow execution failed")
        await queue.put((EVENT_ERROR, exc))
    finally:
        await queue.put((EVENT_DONE, None))


async def _finalize_stream(emitter_token: Any, producer_task: asyncio.Task[Any]) -> str:
    """
    统一收尾逻辑：重置 emitter、取消后台任务、输出结束包。

    为什么必须输出结束包：
    - 前端依赖 `is_end=true` 作为流完成信号；
    - 即使中途报错，也要给出可确定结束点，避免客户端悬挂等待。
    """

    reset_status_emitter(emitter_token)
    if not producer_task.done():
        producer_task.cancel()
        with suppress(asyncio.CancelledError):
            await producer_task
    return build_answer_sse("", True)


def _render_final_sse_responses() -> list[str]:
    """
    取出并序列化所有最终 SSE 响应。

    Returns:
        list[str]: 按优先级排序后的 SSE 文本列表。
    """

    return [
        serialize_sse(response)
        for response in drain_final_sse_responses()
    ]


async def _invoke_answer_completed_callback(
        callback: OnAnswerCompletedCallback | None,
        answer_text: str,
        execution_trace: list[dict[str, Any]] | None,
        token_usage: dict[str, Any] | None,
        has_error: bool,
        thinking_text: str,
) -> None:
    """
    执行“回答完成”回调。

    允许回调为同步函数或异步函数；回调异常会被调用方兜底，不影响主流输出。

    Args:
        callback: 结束回调函数。
        answer_text: 聚合后的完整回答文本。
        execution_trace: 汇总后的节点执行追踪。
        token_usage: 汇总后的消息级 token 使用信息。
        has_error: 本次流式执行是否出现错误。
        thinking_text: 聚合后的完整思考文本。
    """

    if callback is None:
        return

    callback_signature = inspect.signature(callback)
    parameters = list(callback_signature.parameters.values())
    positional_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    accepts_variadic = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    )
    supports_four_args = accepts_variadic or len(positional_parameters) >= 4
    supports_five_args = accepts_variadic or len(positional_parameters) >= 5
    supports_three_args = accepts_variadic or len(positional_parameters) >= 3

    if supports_five_args:
        callback_result = callback(answer_text, execution_trace, token_usage, has_error, thinking_text)
    elif supports_four_args:
        callback_result = callback(answer_text, execution_trace, token_usage, has_error)
    elif supports_three_args:
        callback_result = callback(answer_text, execution_trace, has_error)
    else:
        callback_result = callback(answer_text, execution_trace)
    if inspect.isawaitable(callback_result):
        await callback_result


async def _event_stream(
        *, question: str, config: AssistantStreamConfig
) -> AsyncIterable[str]:
    """
    核心事件流生成器。

    这是通用流式引擎的主循环：
    - 从 workflow 与状态发射器接收事件
    - 按事件类型分发处理
    - 在 done 后 drain 尾事件
    - 必要时输出 fallback answer
    - 最终输出结束包
    """

    state = config.build_initial_state(question)
    runtime_state = StreamRuntimeState(latest_state=state)

    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _event_emitter(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (EVENT_EMITTED, event))

    emitter_token = set_status_emitter(_event_emitter)
    final_response_queue_token = set_final_response_queue()

    # 先把前置事件写入队列，确保“会话创建成功”等通知优先于图执行事件输出。
    for initial_event in config.initial_emitted_events:
        initial_payload = _normalize_initial_event_payload(initial_event)
        if initial_payload is not None:
            queue.put_nowait((EVENT_EMITTED, initial_payload))

    producer_task = asyncio.create_task(
        _produce_workflow_events(
            queue=queue,
            state=state,
            runtime_state=runtime_state,
            config=config,
        )
    )

    try:
        while True:
            event_type, payload = await queue.get()

            event_result = _process_stream_event(
                event_type=event_type,
                payload=payload,
                runtime_state=runtime_state,
                should_stream_token=config.should_stream_token,
                hide_node_types=config.hide_node_types,
                map_exception=config.map_exception,
            )
            for rendered_item in event_result.rendered_events:
                yield rendered_item

            if event_result.should_break:
                # 这里先让出一次调度，再 drain 队列，是为了捕获 done 前后竞态写入的尾事件。
                await asyncio.sleep(0)
                drained_result = await drain_pending_events(
                    queue=queue,
                    runtime_state=runtime_state,
                    should_stream_token=config.should_stream_token,
                    hide_node_types=config.hide_node_types,
                    map_exception=config.map_exception,
                )
                for drained_item in drained_result.rendered_events:
                    yield drained_item
                break

        # 当没有 token 输出且没有错误时，回退到业务侧最终内容提取。
        # 若业务侧返回空字符串，则视为“无兜底内容”，不输出额外 answer 包。
        if not runtime_state.has_emitted_error and not runtime_state.has_streamed_output:
            fallback_text = config.extract_final_content(runtime_state.latest_state)
            if isinstance(fallback_text, str) and fallback_text:
                delta_text = _append_answer_text(runtime_state, fallback_text)
                if delta_text:
                    yield build_answer_sse(delta_text, False)

        for final_response in _render_final_sse_responses():
            yield final_response
    finally:
        try:
            execution_trace = _build_execution_trace_summary(runtime_state.latest_state)
            token_usage = _build_token_usage_summary(runtime_state.latest_state)
            await _invoke_answer_completed_callback(
                config.on_answer_completed,
                runtime_state.aggregated_answer_text,
                execution_trace,
                token_usage,
                runtime_state.has_emitted_error,
                runtime_state.aggregated_thinking_text,
            )
        except Exception as exc:  # pragma: no cover - 防御性兜底
            logger.opt(exception=exc).warning("Assistant stream finalize callback failed")
        reset_final_response_queue(final_response_queue_token)
        end_event = await _finalize_stream(emitter_token, producer_task)
        yield end_event


def create_streaming_response(
        question: str, config: AssistantStreamConfig
) -> StreamingResponse:
    """
    对外统一入口：创建 FastAPI StreamingResponse。

    业务路由仅需组装 `AssistantStreamConfig` 并传入问题文本，
    无需关心队列管理、状态事件透传和收尾细节。
    """

    return StreamingResponse(
        _event_stream(question=question, config=config),
        media_type="text/event-stream",
        headers=config.response_headers,
    )
