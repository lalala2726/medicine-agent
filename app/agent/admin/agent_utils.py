from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.admin.history_utils import history_to_role_dicts
from app.agent.admin.model_policy import (
    normalize_task_difficulty,
    resolve_model_profile,
)
from app.core.assistant_status import emit_thinking_notice
from app.core.llm import create_chat_model
from app.utils.streaming_utils import invoke_with_policy

if TYPE_CHECKING:
    from app.agent.admin.state import AgentState

UNKNOWN_MODEL_NAME = "unknown"


@dataclass(slots=True)
class NodeExecutionResult:
    """
    标准化的 worker 节点执行结果。

    Attributes:
        content: 节点最终输出文本（已做错误标记归一化）。
        status: 执行状态，通常为 `completed` 或 `failed`。
        error: 失败原因，成功时为 `None`。
        model_name: 实际执行所用模型名。
        input_messages: 序列化后的输入消息列表，便于追踪与测试断言。
        tool_calls: 工具调用明细列表，通常来自 `invoke_with_policy` 的诊断信息。
        diagnostics: 诊断信息原始字典（如阈值触发、工具错误统计等）。
        stream_chunks: 流式输出分片列表，便于定位流式阶段问题。
        reasoning_chunks: 深度思考分片列表（当模型开启 thinking 且供应商返回时存在）。
    """

    content: str
    status: str
    error: str | None = None
    model_name: str = UNKNOWN_MODEL_NAME
    input_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    stream_chunks: list[str] = field(default_factory=list)
    reasoning_chunks: list[str] = field(default_factory=list)


def serialize_message(message: Any) -> dict[str, Any]:
    """
    将单条消息对象序列化为统一字典结构。

    该函数用于将 LangChain 的消息对象（或兼容对象）转为可记录、可追踪的
    `dict` 结构，避免各节点重复实现消息序列化逻辑。

    Args:
        message: 任意消息对象。通常为 `HumanMessage`、`AIMessage`、`SystemMessage`
            或其他包含 `type` / `content` 属性的对象。

    Returns:
        dict[str, Any]: 序列化后的消息字典，字段说明如下：
            - `role`: 消息角色名称；优先取 `message.type`，否则回退类名，最终兜底 `unknown`。
            - `content`: 消息正文内容；当原值为 `None` 时返回空字符串。
            - `additional_kwargs`（可选）: 若消息附带额外参数且非空则保留。
            - `name`（可选）: 若消息具名且非空则保留。
    """

    role = str(getattr(message, "type", "") or message.__class__.__name__).strip().lower()
    content = getattr(message, "content", "")
    message_dict: dict[str, Any] = {
        "role": role or "unknown",
        "content": content if content is not None else "",
    }

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict) and additional_kwargs:
        message_dict["additional_kwargs"] = additional_kwargs

    name = getattr(message, "name", None)
    if isinstance(name, str) and name:
        message_dict["name"] = name

    return message_dict


def serialize_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """
    将消息序列批量序列化为字典列表。

    Args:
        messages: 可迭代的消息对象集合，元素会逐个交由 `serialize_message` 处理。

    Returns:
        list[dict[str, Any]]: 序列化后的消息列表，顺序与输入保持一致，
            每个元素的字段定义与 `serialize_message` 返回结构一致。
    """

    return [serialize_message(message) for message in messages]


def _resolve_llm_model_name(llm: Any) -> str:
    """
    从模型对象中提取可识别的模型名称。

    Args:
        llm: 任意模型实例，优先读取 `model_name`，其次 `model` 属性。

    Returns:
        str: 可用模型名；当无法识别时返回 `UNKNOWN_MODEL_NAME`。
    """

    for attr in ("model_name", "model"):
        candidate = getattr(llm, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return UNKNOWN_MODEL_NAME


def build_execution_trace_update(
        *,
        node_name: str,
        model_name: str = UNKNOWN_MODEL_NAME,
        input_messages: list[Any] | None = None,
        output_text: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    构造 `execution_traces` 的增量更新结构。

    Args:
        node_name: 产生该追踪记录的节点名。
        model_name: 执行模型名称。
        input_messages: 输入消息列表（建议传序列化后的结构）。
        output_text: 节点输出文本。
        tool_calls: 工具调用明细列表。

    Returns:
        dict[str, Any]: 可直接合并到图状态的更新字典，格式为：
            `{"execution_traces": [trace_item]}`，其中 `trace_item` 包含节点名、模型名、
            输入消息、输出文本和工具调用明细。
    """

    trace_item = {
        "node_name": str(node_name or UNKNOWN_MODEL_NAME),
        "model_name": str(model_name or UNKNOWN_MODEL_NAME),
        "input_messages": list(input_messages or []),
        "output_text": str(output_text or ""),
        "tool_calls": list(tool_calls or []),
    }
    return {"execution_traces": [trace_item]}


def _resolve_invoke_policy(failure_policy: dict[str, Any] | None) -> dict[str, Any]:
    """
    解析节点失败策略并填充默认值。

    Args:
        failure_policy: 外部传入的失败策略字典，可选。

    Returns:
        dict[str, Any]: 标准化后的策略字典，固定包含：
            - `error_marker_prefix`: 模型错误标记前缀；
            - `tool_error_counting`: 工具错误计数方式；
            - `max_tool_errors`: 最大工具错误阈值。
    """

    policy = dict(failure_policy or {})
    return {
        "error_marker_prefix": str(policy.get("error_marker_prefix") or "__ERROR__:"),
        "tool_error_counting": str(policy.get("tool_error_counting") or "consecutive"),
        "max_tool_errors": int(policy.get("max_tool_errors") or 2),
    }


def _resolve_result_status(
        content: str,
        diagnostics: dict[str, Any] | None,
        *,
        error_marker_prefix: str,
) -> tuple[str, str | None, str]:
    """
    根据模型输出与诊断信息判定节点最终状态。

    Args:
        content: 模型返回文本。
        diagnostics: `invoke_with_policy` 返回的诊断信息。
        error_marker_prefix: 错误标记前缀，用于识别模型显式失败输出。

    Returns:
        tuple[str, str | None, str]: 三元组，依次为：
            1. `status`: `completed` 或 `failed`；
            2. `error`: 失败原因，成功时为 `None`；
            3. `normalized_content`: 归一化后的输出文本。
    """

    normalized_content = str(content or "").strip()
    if normalized_content.startswith(error_marker_prefix):
        marker_reason = normalized_content[len(error_marker_prefix):].strip() or "模型返回错误标记。"
        return "failed", marker_reason, marker_reason

    diagnostics = diagnostics or {}
    if bool(diagnostics.get("threshold_hit")):
        reason = str(diagnostics.get("threshold_reason") or "").strip() or "工具失败次数达到阈值。"
        if not normalized_content:
            normalized_content = reason
        return "failed", reason, normalized_content
    return "completed", None, normalized_content


def execute_tool_node(
        *,
        llm: Any,
        messages: list[Any],
        tools: Sequence[Any],
        enable_stream: bool,
        failure_policy: dict[str, Any] | None,
        fallback_content: str,
        fallback_error: str,
) -> NodeExecutionResult:
    """
    执行带工具的节点调用，并统一错误处理与结果结构。

    Args:
        llm: 模型实例。
        messages: 输入消息列表。
        tools: 可用工具序列。
        enable_stream: 是否启用流式调用。
        failure_policy: 失败策略配置。
        fallback_content: 调用异常时返回给用户的兜底文本。
        fallback_error: 调用异常时写入结果的错误前缀。

    Returns:
        NodeExecutionResult: 标准化执行结果对象，包含最终文本、状态、错误、
            工具调用明细与诊断数据。
    """

    model_name = _resolve_llm_model_name(llm)
    serialized_inputs = serialize_messages(messages)
    invoke_policy = _resolve_invoke_policy(failure_policy)
    marker_prefix = str(invoke_policy["error_marker_prefix"])

    try:
        content, diagnostics = invoke_with_policy(
            llm,
            messages,
            tools=tools,
            enable_stream=enable_stream,
            error_marker_prefix=marker_prefix,
            tool_error_counting=str(invoke_policy["tool_error_counting"]),
            max_tool_errors=int(invoke_policy["max_tool_errors"]),
        )
        step_status, failed_error, normalized_content = _resolve_result_status(
            content,
            diagnostics,
            error_marker_prefix=marker_prefix,
        )
        return NodeExecutionResult(
            content=normalized_content,
            status=step_status,
            error=failed_error,
            model_name=model_name,
            input_messages=serialized_inputs,
            tool_calls=list((diagnostics or {}).get("tool_call_details") or []),
            diagnostics=diagnostics or {},
            stream_chunks=list((diagnostics or {}).get("stream_chunks") or []),
            reasoning_chunks=list((diagnostics or {}).get("reasoning_chunks") or []),
        )
    except Exception as exc:
        return NodeExecutionResult(
            content=fallback_content,
            status="failed",
            error=f"{fallback_error}: {exc}",
            model_name=model_name,
            input_messages=serialized_inputs,
            tool_calls=[],
            diagnostics={},
            stream_chunks=[],
            reasoning_chunks=[],
        )


def build_mode_aware_instruction_payload(state: AgentState) -> dict[str, Any]:
    """
    根据当前路由模式构建 worker instruction 负载。

    规则说明：
    1. 总是写入 `user_input`、`context`、`execution_mode`；
    2. `fast_lane` 下附加 `chat_history`，让直达节点具备历史记忆；
    3. `supervisor_loop` 下仅附加 `directive`，避免重复消耗历史 token。

    Args:
        state: 当前图状态，至少应包含 `routing`、`context`、`messages`、`user_input` 字段。

    Returns:
        dict[str, Any]: 供下游 worker 使用的 instruction 字典，核心字段说明如下：
            - `user_input`: 用户当前轮输入。
            - `context`: 跨节点共享上下文。
            - `execution_mode`: 执行模式（`fast_lane`/`supervisor_loop`/其他）。
            - `task_difficulty`: 当前任务难度（simple/normal/complex）。
            - `chat_history`（可选）: 仅 `fast_lane` 时存在，值为 role/content 历史列表。
            - `directive`（可选）: 仅 `supervisor_loop` 时存在，值为主管下发指令。
    """

    routing = dict(state.get("routing") or {})
    execution_mode = str(routing.get("mode") or "")

    payload: dict[str, Any] = {
        "user_input": state.get("user_input"),
        "context": state.get("context") or {},
        "execution_mode": execution_mode,
        "task_difficulty": normalize_task_difficulty(routing.get("task_difficulty")),
    }

    if execution_mode == "fast_lane":
        payload["chat_history"] = history_to_role_dicts(list(state.get("messages") or []))
    elif execution_mode == "supervisor_loop":
        payload["directive"] = str(routing.get("directive") or "").strip()

    return payload


def build_worker_input_messages(
        system_prompt: str,
        instruction_payload: dict[str, Any],
) -> list[Any]:
    """
    构建标准 worker 模型输入消息。

    Args:
        system_prompt: 节点系统提示词。
        instruction_payload: 通过 `build_mode_aware_instruction_payload` 生成的 instruction 字典。

    Returns:
        list[Any]: 两条消息组成的列表：
            1. `SystemMessage(system_prompt)`；
            2. `HumanMessage(JSON instruction)`，JSON 序列化保持中文可读。
    """

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(instruction_payload, ensure_ascii=False, default=str)),
    ]


def run_standard_tool_worker(
        *,
        state: AgentState,
        node_name: str,
        result_key: str,
        system_prompt: str,
        tools: Sequence[Any],
        fallback_content: str,
        fallback_error: str,
        model_name: str | None = None,
        think: bool | None = None,
        enable_stream: bool = True,
        failure_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    执行标准化工具型 worker 节点并返回统一状态更新。

    该封装统一处理以下流程：
    1. 组装模式化 instruction 负载；
    2. 构造模型输入消息；
    3. 调用 LLM + tools 执行；
    4. 生成与既有节点一致的 `results/context/messages/execution_traces/errors` 更新结构。

    Args:
        state: 当前图状态。
        node_name: 节点名（例如 `order_agent`、`product_agent`）。
        result_key: 写入 `results` 的键名（例如 `order`、`product`）。
        system_prompt: 当前节点系统提示词。
        tools: 该节点可调用的工具集合。
        fallback_content: 执行异常时返回给用户的兜底文本。
        fallback_error: 执行异常时记录到 `errors` 的错误前缀。
        model_name: 可选模型覆盖。未传时按 `routing.task_difficulty` 自动选择模型。
        think: 可选深度思考覆盖。未传时按 `routing.task_difficulty` 自动选择。
        enable_stream: 是否开启流式工具调用，默认开启。
        failure_policy: 失败策略配置，会透传给 `execute_tool_node`。

    Returns:
        dict[str, Any]: 可直接用于 LangGraph 状态合并的增量更新字典，字段语义如下：
            - `results[result_key]`: 当前节点输出内容与 `is_end` 标记。
            - `context`: 追加 `agent_outputs`、`last_agent`、提取到的 ID 等共享上下文。
            - `messages`: 追加一条 `AIMessage` 作为本节点对话输出。
            - `execution_traces`: 记录输入消息、模型名、工具调用与输出文本。
            - `errors`（可选）: 当节点失败时写入错误信息。
    """

    from app.agent.admin.node.common import build_worker_update

    routing = dict(state.get("routing") or {})
    task_difficulty = normalize_task_difficulty(routing.get("task_difficulty"))
    profile = resolve_model_profile(task_difficulty)
    selected_model = str(model_name or profile.get("model") or "qwen-plus")
    think_enabled = bool(profile.get("think")) if think is None else bool(think)

    instruction_payload = build_mode_aware_instruction_payload(state)
    input_messages = build_worker_input_messages(system_prompt, instruction_payload)

    llm = create_chat_model(
        model=selected_model,
        think=think_enabled,
    )
    execution_result = execute_tool_node(
        llm=llm,
        messages=input_messages,
        tools=tools,
        enable_stream=enable_stream,
        failure_policy=failure_policy or {},
        fallback_content=fallback_content,
        fallback_error=fallback_error,
    )

    if think_enabled and execution_result.reasoning_chunks:
        emit_thinking_notice(
            node=node_name,
            state="thinking_start",
            meta={
                "model": selected_model,
                "task_difficulty": task_difficulty,
            },
        )
        for chunk in execution_result.reasoning_chunks:
            emit_thinking_notice(
                node=node_name,
                state="thinking_delta",
                text=chunk,
                meta={
                    "model": selected_model,
                    "task_difficulty": task_difficulty,
                },
            )
        emit_thinking_notice(
            node=node_name,
            state="thinking_end",
            meta={
                "model": selected_model,
                "task_difficulty": task_difficulty,
            },
        )

    return build_worker_update(
        state=state,
        node_name=node_name,
        result_key=result_key,
        content=execution_result.content,
        status=execution_result.status,
        model_name=execution_result.model_name,
        input_messages=serialize_messages(input_messages),
        tool_calls=execution_result.tool_calls,
        error=execution_result.error,
    )


__all__ = [
    "UNKNOWN_MODEL_NAME",
    "NodeExecutionResult",
    "serialize_message",
    "serialize_messages",
    "build_execution_trace_update",
    "execute_tool_node",
    "build_mode_aware_instruction_payload",
    "build_worker_input_messages",
    "run_standard_tool_worker",
]
