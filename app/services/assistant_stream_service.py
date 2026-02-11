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
import json
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Callable

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.core.assistant_status import reset_status_emitter, set_status_emitter
from app.schemas.sse_response import AssistantResponse, Content, MessageType
from app.utils.streaming_utils import extract_text

StreamEvent = tuple[str, Any]
GraphEventPayload = tuple[str, Any]

EVENT_EMITTED = "emitted"
EVENT_GRAPH = "graph"
EVENT_ERROR = "error"
EVENT_DONE = "done"

GRAPH_MODE_MESSAGES = "messages"
GRAPH_MODE_VALUES = "values"


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

    normalized_type = str(raw_type or MessageType.STATUS.value)
    try:
        return MessageType(normalized_type)
    except ValueError:
        return MessageType.STATUS


def build_emitted_sse(
        event_payload: dict[str, Any], hide_node_types: set[MessageType]
) -> str | None:
    """
    将状态发射器事件（status/function_call/tool_response）转换为 SSE 文本。

    Args:
        event_payload: 发射器推送的原始事件。
        hide_node_types: 需要隐藏 `node` 字段的事件类型集合。

    Returns:
        str | None: 可序列化时返回 SSE 文本；结构不合法时返回 None 并忽略该事件。

    说明：
    - `function_call` 等事件按配置隐藏 `node`，避免把“工具节点内部标识”暴露给前端。
    - 如果事件结构不合法（例如 content 不是字典），选择忽略而非抛错，
      保障主流式链路稳定。
    """

    content = event_payload.get("content")
    if not isinstance(content, dict):
        return None

    message_type = _resolve_message_type(event_payload.get("type"))

    node = content.get("node")
    if message_type in hide_node_types:
        node = None

    payload = AssistantResponse(
        content=Content(
            text=content.get("text"),
            node=node,
            state=content.get("state"),
            message=content.get("message"),
            result=content.get("result"),
            name=content.get("name"),
            arguments=content.get("arguments"),
        ),
        type=message_type,
    )
    return serialize_sse(payload)


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

    token_text = extract_text(message_chunk)
    if not token_text:
        return result

    runtime_state.has_streamed_output = True
    result.rendered_events.append(build_answer_sse(token_text, False))
    return result


def _process_graph_values_event(chunk: Any, runtime_state: StreamRuntimeState) -> None:
    """
    处理 graph `values` 事件。

    values 事件用于更新最新状态快照，不直接输出给前端。
    """

    if isinstance(chunk, dict):
        runtime_state.latest_state = chunk


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
        rendered = build_emitted_sse(payload, hide_node_types)
        result = EventProcessResult()
        if rendered:
            result.rendered_events.append(rendered)
        return result

    if event_type == EVENT_GRAPH:
        return _process_graph_event(payload, runtime_state, should_stream_token)

    if event_type == EVENT_ERROR:
        runtime_state.has_emitted_error = True
        result = EventProcessResult()
        message = map_exception(payload)
        result.rendered_events.append(build_answer_sse(message, False))
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

        # 当没有 token 输出且没有错误时，回退到业务侧最终内容提取，避免“空响应”。
        if not runtime_state.has_emitted_error and not runtime_state.has_streamed_output:
            fallback_text = config.extract_final_content(runtime_state.latest_state)
            yield build_answer_sse(fallback_text, False)
    finally:
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
