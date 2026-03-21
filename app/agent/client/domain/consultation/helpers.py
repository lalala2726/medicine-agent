from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from app.agent.client.domain.consultation.schema import ConsultationQuestionSchema
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COMPLETED,
    ConsultationInterruptPayload,
    ConsultationState,
)
from app.agent.client.state import ExecutionTraceState, ToolCallTraceState
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.schemas.sse_response import AssistantResponse, Card, MessageType
from app.services.token_usage_service import (
    append_trace_and_refresh_token_usage,
    build_token_usage_from_execution_traces,
)
from app.utils.prompt_utils import append_current_time_to_prompt

# consultation 节点兜底的安抚文本。
DEFAULT_COMFORT_TEXT = "先别太担心，我们先按你已经说到的情况一步步判断。"
# consultation 节点兜底的阶段性分析文本。
DEFAULT_QUESTION_REPLY_TEXT = "根据你现在提供的信息，我们大致更偏向常见轻症方向，但还需要再确认一个关键点。"
# consultation 节点兜底的提问标题。
DEFAULT_QUESTION_TEXT = "为了更准确判断，我还想确认一个症状。"
# consultation 节点兜底的追问选项。
DEFAULT_QUESTION_OPTIONS = ["没有", "轻微", "明显", "不确定"]
# consultation 中断回放兜底文本。
DEFAULT_REPLY_TEXT = DEFAULT_QUESTION_REPLY_TEXT
# consultation 节点默认温度配置。
DEFAULT_TEMPERATURE = 0.2
# consultation 安抚节点模型槽位。
CONSULTATION_COMFORT_MODEL_SLOT = AgentChatModelSlot.CLIENT_CONSULTATION_COMFORT
# consultation 追问节点模型槽位。
CONSULTATION_QUESTION_MODEL_SLOT = AgentChatModelSlot.CLIENT_CONSULTATION_QUESTION
# consultation 最终诊断节点模型槽位。
CONSULTATION_FINAL_DIAGNOSIS_MODEL_SLOT = AgentChatModelSlot.CLIENT_CONSULTATION_FINAL_DIAGNOSIS
# consultation 子节点统一关闭深度思考。
CONSULTATION_AGENT_THINK_ENABLED = False
# consultation 统一分片时单个分片最大长度。
CONSULTATION_STREAM_CHUNK_MAX_LENGTH = 80
# consultation 统一分片时优先按更短软长度切分。
CONSULTATION_STREAM_SOFT_CHUNK_LENGTH = 26
# consultation 优先视为句尾的标点集合。
CONSULTATION_STREAM_SENTENCE_ENDINGS = frozenset({"。", "！", "？", "；"})
# consultation 软切分优先使用的标点集合。
CONSULTATION_STREAM_SOFT_BREAKS = frozenset({"，", "、", "：", "；"})
# consultation 子图固定使用的 checkpoint namespace。
CONSULTATION_CHECKPOINT_NAMESPACE = "client_consultation"
# consultation interrupt payload 类型。
CONSULTATION_INTERRUPT_KIND = "consultation_question"
# consultation 专用追问卡片类型标识。
CONSULTATION_FOLLOWUP_CARD_TYPE = "consultation-followup-card"
# consultation 追问卡片默认选择模式。
CONSULTATION_FOLLOWUP_SELECTION_MODE = "multiple"
# consultation 追问卡片默认提交文案。
CONSULTATION_FOLLOWUP_SUBMIT_TEXT = "发送"
# consultation 追问卡片默认自定义输入占位文案。
CONSULTATION_FOLLOWUP_CUSTOM_INPUT_PLACEHOLDER = "补充症状或其他感受"

def resolve_payload_text(raw_payload: Any) -> str:
    """
    功能描述：
        从 agent payload 中提取最后一条 AI 文本。

    参数说明：
        raw_payload (Any): agent payload 原始数据。

    返回值：
        str: 提取到的文本；没有有效文本时返回空串。

    异常说明：
        无。
    """

    if isinstance(raw_payload, Mapping):
        raw_messages = raw_payload.get("messages")
        if isinstance(raw_messages, list) and raw_messages:
            last_message = raw_messages[-1]
            raw_content = getattr(last_message, "content", None)
            if isinstance(raw_content, str):
                return raw_content.strip()
    return ""


def parse_json_text(raw_payload: Any) -> dict[str, Any] | None:
    """
    功能描述：
        解析节点输出中的 JSON 文本。

    参数说明：
        raw_payload (Any): agent payload 原始数据。

    返回值：
        dict[str, Any] | None: 解析成功返回 JSON 对象，否则返回 `None`。

    异常说明：
        无。
    """

    payload_text = resolve_payload_text(raw_payload)
    if not payload_text:
        return None

    try:
        parsed_json = json.loads(payload_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed_json, dict):
        return None
    return parsed_json


def resolve_question_result(raw_payload: Any) -> ConsultationQuestionSchema:
    """
    功能描述：
        解析 consultation 追问节点结构化输出。

    参数说明：
        raw_payload (Any): 追问节点原始 payload。

    返回值：
        ConsultationQuestionSchema: 解析成功或兜底后的结构化结果。

    异常说明：
        无；解析失败时返回兜底结果。
    """

    parsed_json = parse_json_text(raw_payload)
    if parsed_json is None:
        return ConsultationQuestionSchema(
            diagnosis_ready=False,
            question_reply_text=DEFAULT_QUESTION_REPLY_TEXT,
            question_text=DEFAULT_QUESTION_TEXT,
            options=list(DEFAULT_QUESTION_OPTIONS),
        )

    try:
        return ConsultationQuestionSchema.model_validate(parsed_json)
    except ValidationError:
        return ConsultationQuestionSchema(
            diagnosis_ready=False,
            question_reply_text=DEFAULT_QUESTION_REPLY_TEXT,
            question_text=DEFAULT_QUESTION_TEXT,
            options=list(DEFAULT_QUESTION_OPTIONS),
        )


def build_trace_item(
        *,
        node_name: str,
        llm_model_name: str,
        output_text: str,
        llm_usage_complete: bool = True,
        llm_token_usage: dict[str, Any] | None = None,
        tool_calls: list[ToolCallTraceState] | None = None,
        node_context: dict[str, Any] | None = None,
) -> ExecutionTraceState:
    """
    功能描述：
        构造 consultation 节点 execution trace。

    参数说明：
        node_name (str): 节点名称。
        llm_model_name (str): 模型名称。
        output_text (str): 节点输出文本。
        llm_usage_complete (bool): LLM usage 是否完整。
        llm_token_usage (dict[str, Any] | None): 节点 token 使用量。
        tool_calls (list[ToolCallTraceState] | None): 工具调用明细。
        node_context (dict[str, Any] | None): 节点上下文。

    返回值：
        ExecutionTraceState: 统一节点追踪结构。

    异常说明：
        无。
    """

    return ExecutionTraceState(
        sequence=0,
        node_name=node_name,
        model_name=llm_model_name or "unknown",
        status="success",
        output_text=output_text.strip(),
        llm_usage_complete=llm_usage_complete,
        llm_token_usage=llm_token_usage,
        tool_calls=list(tool_calls or []),
        node_context=node_context,
    )


def _resequence_execution_traces(
        execution_traces: list[ExecutionTraceState],
) -> list[ExecutionTraceState]:
    """
    功能描述：
        重新为 execution_traces 分配顺序编号。

    参数说明：
        execution_traces (list[ExecutionTraceState]): 原始节点执行轨迹列表。

    返回值：
        list[ExecutionTraceState]: 顺序编号从 1 开始的新列表。

    异常说明：
        无。
    """

    normalized_traces: list[ExecutionTraceState] = []
    for index, raw_trace in enumerate(execution_traces, start=1):
        trace_payload = dict(raw_trace)
        trace_payload["sequence"] = index
        normalized_traces.append(ExecutionTraceState(**trace_payload))
    return normalized_traces


def append_trace_to_state(
        *,
        state: ConsultationState,
        trace_item: ExecutionTraceState,
) -> tuple[list[ExecutionTraceState], dict[str, Any] | None]:
    """
    功能描述：
        追加单条 consultation trace，并同步刷新 token_usage。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。
        trace_item (ExecutionTraceState): 新增节点 trace。

    返回值：
        tuple[list[ExecutionTraceState], dict[str, Any] | None]:
            - 新的 execution_traces；
            - 新的 token_usage。

    异常说明：
        无。
    """

    raw_traces, _ = append_trace_and_refresh_token_usage(
        execution_traces=state.get("execution_traces"),
        trace_item=trace_item,
    )
    resequenced_traces = _resequence_execution_traces(raw_traces)
    token_usage = build_token_usage_from_execution_traces(resequenced_traces)
    return resequenced_traces, token_usage


def merge_parallel_round_traces(
        *,
        state: ConsultationState,
) -> tuple[list[ExecutionTraceState], dict[str, Any] | None]:
    """
    功能描述：
        合并当前 collecting 轮次的 comfort/question 两条并行 trace。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        tuple[list[ExecutionTraceState], dict[str, Any] | None]:
            - 合并后的 execution_traces；
            - 对应的 token_usage。

    异常说明：
        无。
    """

    merged_traces = list(state.get("execution_traces") or [])
    token_usage = build_token_usage_from_execution_traces(merged_traces)

    for trace_key in ("comfort_trace", "question_trace"):
        raw_trace = state.get(trace_key)
        if not isinstance(raw_trace, Mapping):
            continue

        merged_traces, token_usage = append_trace_to_state(
            state=ConsultationState(execution_traces=merged_traces),
            trace_item=ExecutionTraceState(**dict(raw_trace)),
        )

    return merged_traces, token_usage


def build_llm_agent(
        *,
        state: ConsultationState,
        slot: AgentChatModelSlot,
        prompt_text: str,
        temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[Any, str]:
    """
    功能描述：
        构造 consultation 节点使用的 agent。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。
        slot (AgentChatModelSlot): 当前节点绑定的客户端助手模型槽位。
        prompt_text (str): 节点系统提示词。
        temperature (float): 模型温度。

    返回值：
        tuple[Any, str]:
            - agent 实例；
            - 模型名称。

    异常说明：
        无。
    """

    _ = state
    llm = create_agent_chat_llm(
        slot=slot,
        temperature=temperature,
        think=CONSULTATION_AGENT_THINK_ENABLED,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip() or "unknown"
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(content=append_current_time_to_prompt(prompt_text)),
        middleware=[
            BasePromptMiddleware(base_prompt_file="client/_client_base_prompt.md"),
        ],
    )
    return agent, llm_model_name


def build_text_result(*parts: str) -> str:
    """
    功能描述：
        按固定顺序拼接用户可见文本。

    参数说明：
        *parts (str): 待拼接文本片段。

    返回值：
        str: 清理空白并按双换行拼接后的文本。

    异常说明：
        无。
    """

    normalized_parts = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    return "\n\n".join(normalized_parts)


def resolve_natural_language_text(
        *,
        trace_text: str,
        fallback_text: str,
        default_text: str,
) -> str:
    """
    功能描述：
        为自然语言节点提取最终展示文本，并提供统一兜底。

    参数说明：
        trace_text (str): 从 trace 中提取的文本。
        fallback_text (str): 从调用结果中提取的兜底文本。
        default_text (str): 最终兜底文本。

    返回值：
        str: 可直接展示给用户的节点文本。

    异常说明：
        无。
    """

    normalized_trace_text = str(trace_text or "").strip()
    if normalized_trace_text:
        return normalized_trace_text

    normalized_fallback_text = str(fallback_text or "").strip()
    if normalized_fallback_text:
        return normalized_fallback_text

    return str(default_text or "").strip()


def _split_stream_paragraph(
        paragraph_text: str,
        *,
        max_chunk_length: int,
) -> list[str]:
    """
    功能描述：
        优先按句末标点拆分单个段落，再对超长句子做软切分。

    参数说明：
        paragraph_text (str): 单个段落文本。
        max_chunk_length (int): 分片的硬上限。

    返回值：
        list[str]: 保持原文顺序的段落分片列表。

    异常说明：
        无。
    """

    normalized_text = str(paragraph_text or "")
    if not normalized_text:
        return []

    sentences: list[str] = []
    current_sentence: list[str] = []
    for character in normalized_text:
        current_sentence.append(character)
        if character in CONSULTATION_STREAM_SENTENCE_ENDINGS:
            sentences.append("".join(current_sentence))
            current_sentence = []

    if current_sentence:
        sentences.append("".join(current_sentence))

    if not sentences:
        sentences = [normalized_text]

    stream_chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= CONSULTATION_STREAM_SOFT_CHUNK_LENGTH:
            stream_chunks.append(sentence)
            continue

        remaining_text = sentence
        while remaining_text:
            if len(remaining_text) <= CONSULTATION_STREAM_SOFT_CHUNK_LENGTH:
                stream_chunks.append(remaining_text)
                break

            search_limit = min(len(remaining_text), max_chunk_length)
            soft_limit = min(search_limit, CONSULTATION_STREAM_SOFT_CHUNK_LENGTH)
            split_index = 0
            for index in range(soft_limit - 1, -1, -1):
                if remaining_text[index] in CONSULTATION_STREAM_SOFT_BREAKS:
                    split_index = index + 1
                    break

            if split_index <= 0:
                split_index = soft_limit

            stream_chunks.append(remaining_text[:split_index])
            remaining_text = remaining_text[split_index:]

    return stream_chunks


def split_consultation_stream_text(
        text: str,
        *,
        max_chunk_length: int = CONSULTATION_STREAM_CHUNK_MAX_LENGTH,
) -> list[str]:
    """
    功能描述：
        按 consultation 统一规则切分文本分片，并保持原文顺序不变。

    参数说明：
        text (str): 待切分的完整文本。
        max_chunk_length (int): 单个分片硬上限。

    返回值：
        list[str]: 实际切分出的文本分片。

    异常说明：
        无。
    """

    normalized_text = str(text or "").strip()
    if not normalized_text:
        return []

    stream_chunks: list[str] = []
    paragraphs = normalized_text.split("\n\n")
    for paragraph_index, paragraph_text in enumerate(paragraphs):
        paragraph_chunks = _split_stream_paragraph(
            paragraph_text,
            max_chunk_length=max_chunk_length,
        )
        if paragraph_index < len(paragraphs) - 1:
            if paragraph_chunks:
                paragraph_chunks[-1] = paragraph_chunks[-1] + "\n\n"
            else:
                paragraph_chunks = ["\n\n"]
        stream_chunks.extend(chunk for chunk in paragraph_chunks if chunk)

    return stream_chunks


def build_interrupt_payload(
        *,
        reply_text: str,
        question_text: str,
        options: list[str],
) -> ConsultationInterruptPayload:
    """
    功能描述：
        构造 consultation question interrupt payload。

    参数说明：
        reply_text (str): 展示给用户的阶段性分析文本。
        question_text (str): 展示给用户的追问标题。
        options (list[str]): 选择卡片选项列表。

    返回值：
        ConsultationInterruptPayload: 标准化的中断负载。

    异常说明：
        无。
    """

    normalized_options = [str(item).strip() for item in options if str(item).strip()]
    if len(normalized_options) < 2:
        normalized_options = list(DEFAULT_QUESTION_OPTIONS)

    return ConsultationInterruptPayload(
        kind=CONSULTATION_INTERRUPT_KIND,
        reply_text=reply_text.strip() or DEFAULT_REPLY_TEXT,
        question_text=question_text.strip() or DEFAULT_QUESTION_TEXT,
        options=normalized_options[:4],
    )


def resolve_interrupt_payload(state: Mapping[str, Any]) -> ConsultationInterruptPayload | None:
    """
    功能描述：
        从 graph 最新状态中提取 consultation interrupt payload。

    参数说明：
        state (Mapping[str, Any]): 最新 workflow state。

    返回值：
        ConsultationInterruptPayload | None: 命中时返回中断负载，否则返回 `None`。

    异常说明：
        无。
    """

    raw_interrupts = state.get("__interrupt__")
    if not isinstance(raw_interrupts, (list, tuple)) or not raw_interrupts:
        return None

    first_interrupt = raw_interrupts[0]
    raw_value = getattr(first_interrupt, "value", None)
    if not isinstance(raw_value, dict):
        return None

    kind = str(raw_value.get("kind") or "").strip()
    if kind != CONSULTATION_INTERRUPT_KIND:
        return None

    question_text = str(raw_value.get("question_text") or "").strip() or DEFAULT_QUESTION_TEXT
    reply_text = str(
        raw_value.get("reply_text")
        or raw_value.get("question_text")
        or ""
    ).strip() or DEFAULT_REPLY_TEXT
    raw_options = raw_value.get("options")
    options = [
        str(item).strip()
        for item in (raw_options if isinstance(raw_options, list) else DEFAULT_QUESTION_OPTIONS)
        if str(item).strip()
    ]
    if len(options) < 2:
        options = list(DEFAULT_QUESTION_OPTIONS)

    return ConsultationInterruptPayload(
        kind=kind,
        reply_text=reply_text,
        question_text=question_text,
        options=options[:4],
    )


def build_consultation_followup_card_response(
        *,
        title: str,
        description: str,
        options: list[str],
        message_uuid: str | None = None,
) -> AssistantResponse:
    """
    功能描述：
        构造 consultation 追问即时发送的专用追问卡片响应。

    参数说明：
        title (str): 卡片标题。
        description (str): 卡片说明文本。
        options (list[str]): 卡片选项列表。
        message_uuid (str | None): 前端用于拆分追问卡片块的逻辑消息 UUID。

    返回值：
        AssistantResponse: 可直接推送给前端并持久化的卡片响应。

    异常说明：
        无。
    """

    normalized_title = str(title or "").strip() or DEFAULT_QUESTION_TEXT
    normalized_description = str(description or "").strip() or DEFAULT_QUESTION_REPLY_TEXT
    normalized_options = [
        str(item).strip()
        for item in options
        if str(item).strip()
    ]
    if len(normalized_options) < 2:
        normalized_options = list(DEFAULT_QUESTION_OPTIONS)

    followup_message_uuid = str(message_uuid or "").strip() or str(uuid.uuid4())
    return AssistantResponse(
        type=MessageType.CARD,
        card=Card(
            type=CONSULTATION_FOLLOWUP_CARD_TYPE,
            data={
                "title": normalized_title,
                "description": normalized_description,
                "options": normalized_options[:4],
                "selectionMode": CONSULTATION_FOLLOWUP_SELECTION_MODE,
                "submitText": CONSULTATION_FOLLOWUP_SUBMIT_TEXT,
                "allowCustomInput": True,
                "customInputPlaceholder": CONSULTATION_FOLLOWUP_CUSTOM_INPUT_PLACEHOLDER,
            },
        ),
        meta={
            "card_uuid": str(uuid.uuid4()),
            "persist_card": True,
            "message_uuid": followup_message_uuid,
        },
    )


def resolve_resume_text(resume_value: Any) -> str:
    """
    功能描述：
        将 interrupt resume 值归一化为用户文本。

    参数说明：
        resume_value (Any): `Command(resume=...)` 传入的原始值。

    返回值：
        str: 标准化后的用户回答文本。

    异常说明：
        无。
    """

    if isinstance(resume_value, Mapping):
        for key in ("question", "text", "value", "content"):
            normalized = str(resume_value.get(key) or "").strip()
            if normalized:
                return normalized
    return str(resume_value or "").strip()


def append_resume_messages(
        *,
        state: ConsultationState,
        ai_reply_text: str,
        resume_text: str,
) -> list[Any]:
    """
    功能描述：
        将“上一轮 AI 文本 + 本轮用户回答”追加回 consultation history_messages。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。
        ai_reply_text (str): 上一轮 AI 对用户展示的完整文本。
        resume_text (str): 当前用户回答文本。

    返回值：
        list[Any]: 追加后的 history_messages 列表。

    异常说明：
        无。
    """

    history_messages = list(state.get("history_messages") or [])
    normalized_ai_reply_text = str(ai_reply_text or "").strip()
    normalized_resume_text = str(resume_text or "").strip()

    if normalized_ai_reply_text:
        history_messages.append(AIMessage(content=normalized_ai_reply_text))
    if normalized_resume_text:
        history_messages.append(HumanMessage(content=normalized_resume_text))

    return history_messages


def build_consultation_graph_config(config: RunnableConfig | None) -> RunnableConfig:
    """
    功能描述：
        基于上游 RunnableConfig 构造 consultation 子图专用 config。

    参数说明：
        config (RunnableConfig | None): 上游 workflow config。

    返回值：
        RunnableConfig: consultation 子图 config，固定带 `thread_id + checkpoint_ns`。

    异常说明：
        ValueError: 缺失 thread_id 时抛出。
    """

    if not isinstance(config, dict):
        raise ValueError("consultation 子图需要有效的 RunnableConfig")

    configurable = dict(config.get("configurable") or {})
    thread_id = str(configurable.get("thread_id") or "").strip()
    if not thread_id:
        raise ValueError("consultation 子图缺少 thread_id")

    configurable["thread_id"] = thread_id
    configurable["checkpoint_ns"] = CONSULTATION_CHECKPOINT_NAMESPACE

    consultation_config = dict(config)
    consultation_config["configurable"] = configurable
    return consultation_config


__all__ = [
    "CONSULTATION_CHECKPOINT_NAMESPACE",
    "CONSULTATION_FOLLOWUP_CARD_TYPE",
    "CONSULTATION_FOLLOWUP_CUSTOM_INPUT_PLACEHOLDER",
    "CONSULTATION_FOLLOWUP_SELECTION_MODE",
    "CONSULTATION_FOLLOWUP_SUBMIT_TEXT",
    "CONSULTATION_COMFORT_MODEL_SLOT",
    "CONSULTATION_FINAL_DIAGNOSIS_MODEL_SLOT",
    "CONSULTATION_QUESTION_MODEL_SLOT",
    "CONSULTATION_AGENT_THINK_ENABLED",
    "CONSULTATION_INTERRUPT_KIND",
    "CONSULTATION_STREAM_CHUNK_MAX_LENGTH",
    "CONSULTATION_STREAM_SENTENCE_ENDINGS",
    "CONSULTATION_STREAM_SOFT_BREAKS",
    "CONSULTATION_STREAM_SOFT_CHUNK_LENGTH",
    "DEFAULT_COMFORT_TEXT",
    "DEFAULT_QUESTION_OPTIONS",
    "DEFAULT_QUESTION_REPLY_TEXT",
    "DEFAULT_QUESTION_TEXT",
    "DEFAULT_REPLY_TEXT",
    "DEFAULT_TEMPERATURE",
    "append_resume_messages",
    "append_trace_to_state",
    "build_consultation_graph_config",
    "build_consultation_followup_card_response",
    "build_interrupt_payload",
    "build_llm_agent",
    "build_text_result",
    "build_trace_item",
    "merge_parallel_round_traces",
    "parse_json_text",
    "resolve_interrupt_payload",
    "resolve_natural_language_text",
    "resolve_payload_text",
    "resolve_question_result",
    "resolve_resume_text",
    "split_consultation_stream_text",
]
