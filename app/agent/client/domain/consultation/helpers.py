from __future__ import annotations

import asyncio
import contextvars
import json
from collections.abc import Mapping
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from app.agent.client.domain.consultation.schema import (
    ConsultationFinalDiagnosisSchema,
    ConsultationQuestionSchema,
)
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COMPLETED,
    ConsultationInterruptPayload,
    ConsultationState,
)
from app.agent.client.domain.tools.card_tools import build_card_response
from app.agent.client.domain.tools.schema import SendSelectionCardRequest
from app.agent.client.state import ExecutionTraceState, ToolCallTraceState
from app.core.agent.agent_event_bus import emit_answer_delta, has_status_emitter
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.schemas.sse_response import AssistantResponse, Card
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
# consultation 最终诊断节点兜底文本。
DEFAULT_DIAGNOSIS_TEXT = "结合你目前提供的信息，更像是常见轻症方向；如果症状持续加重，请及时线下就医。"
# consultation 中断回放兜底文本。
DEFAULT_REPLY_TEXT = DEFAULT_QUESTION_REPLY_TEXT
# consultation 最终推荐商品数量上限。
MAX_RECOMMENDED_PRODUCTS = 3
# consultation 购买确认卡默认数量。
DEFAULT_PURCHASE_QUANTITY = 1
# consultation 节点默认温度配置。
DEFAULT_TEMPERATURE = 0.2
# consultation 统一分片时单个分片最大长度。
CONSULTATION_STREAM_CHUNK_MAX_LENGTH = 80
# consultation 统一分片时优先按更短软长度切分。
CONSULTATION_STREAM_SOFT_CHUNK_LENGTH = 26
# consultation 优先视为句尾的标点集合。
CONSULTATION_STREAM_SENTENCE_ENDINGS = frozenset({"。", "！", "？", "；"})
# consultation 软切分优先使用的标点集合。
CONSULTATION_STREAM_SOFT_BREAKS = frozenset({"，", "、", "：", "；"})
# consultation 分片发送之间的最小让渡时间。
CONSULTATION_STREAM_EMIT_INTERVAL_SECONDS = 0.015
# consultation 子图固定使用的 checkpoint namespace。
CONSULTATION_CHECKPOINT_NAMESPACE = "client_consultation"
# consultation interrupt payload 类型。
CONSULTATION_INTERRUPT_KIND = "consultation_question"
# consultation 选择卡类型标识。
CONSULTATION_SELECTION_CARD_TYPE = "selection-card"
# consultation 子图允许的 task_difficulty 到模型槽位映射。
CONSULTATION_MODEL_SLOT_MAP: dict[str, AgentChatModelSlot] = {
    "normal": AgentChatModelSlot.BUSINESS_SIMPLE,
    "high": AgentChatModelSlot.BUSINESS_COMPLEX,
}


def run_async_safely(coro: Any) -> Any:
    """
    功能描述：
        在同步 LangGraph 节点中安全执行异步协程。

    参数说明：
        coro (Any): 待执行的协程对象。

    返回值：
        Any: 协程最终返回值。

    异常说明：
        无；协程异常会继续向上抛出。
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        current_context = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(current_context.run, asyncio.run, coro).result()

    return asyncio.run(coro)


def invoke_runnable(tool_object: Any, payload: dict[str, Any]) -> Any:
    """
    功能描述：
        执行工具对象，优先走异步调用。

    参数说明：
        tool_object (Any): 具备 `invoke/ainvoke` 能力的工具对象。
        payload (dict[str, Any]): 工具调用入参。

    返回值：
        Any: 工具返回结果。

    异常说明：
        TypeError: 工具对象不支持 `invoke/ainvoke` 时抛出。
    """

    ainvoke = getattr(tool_object, "ainvoke", None)
    if callable(ainvoke):
        return run_async_safely(ainvoke(payload))

    invoke = getattr(tool_object, "invoke", None)
    if callable(invoke):
        return invoke(payload)

    raise TypeError(f"tool_object {tool_object!r} does not support invoke/ainvoke")


def resolve_consultation_model_slot(state: ConsultationState) -> AgentChatModelSlot:
    """
    功能描述：
        根据 consultation 的 task_difficulty 选择模型槽位。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。

    返回值：
        AgentChatModelSlot: 最终选中的模型槽位。

    异常说明：
        无。
    """

    task_difficulty = str(state.get("task_difficulty") or "").strip().lower()
    return CONSULTATION_MODEL_SLOT_MAP.get(task_difficulty, AgentChatModelSlot.BUSINESS_SIMPLE)


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


def resolve_final_diagnosis_result(raw_payload: Any) -> ConsultationFinalDiagnosisSchema:
    """
    功能描述：
        解析 consultation 最终诊断节点结构化输出。

    参数说明：
        raw_payload (Any): 最终诊断节点原始 payload。

    返回值：
        ConsultationFinalDiagnosisSchema: 解析成功或兜底后的结构化结果。

    异常说明：
        无；解析失败时返回兜底结果。
    """

    parsed_json = parse_json_text(raw_payload)
    if parsed_json is None:
        return ConsultationFinalDiagnosisSchema(
            diagnosis_text=DEFAULT_DIAGNOSIS_TEXT,
            should_recommend_products=False,
            product_keyword=None,
            product_usage=None,
        )

    try:
        return ConsultationFinalDiagnosisSchema.model_validate(parsed_json)
    except ValidationError:
        return ConsultationFinalDiagnosisSchema(
            diagnosis_text=DEFAULT_DIAGNOSIS_TEXT,
            should_recommend_products=False,
            product_keyword=None,
            product_usage=None,
        )


def build_tool_trace(
        *,
        tool_name: str,
        tool_input: dict[str, Any],
) -> ToolCallTraceState:
    """
    功能描述：
        构造手工工具调用追踪结构。

    参数说明：
        tool_name (str): 工具名称。
        tool_input (dict[str, Any]): 工具输入参数。

    返回值：
        ToolCallTraceState: 统一工具追踪结构。

    异常说明：
        无。
    """

    return ToolCallTraceState(
        tool_name=tool_name,
        tool_call_id=None,
        tool_input=tool_input,
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
        prompt_text: str,
        temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[Any, str]:
    """
    功能描述：
        构造 consultation 节点使用的 agent。

    参数说明：
        state (ConsultationState): 当前 consultation 状态。
        prompt_text (str): 节点系统提示词。
        temperature (float): 模型温度。

    返回值：
        tuple[Any, str]:
            - agent 实例；
            - 模型名称。

    异常说明：
        无。
    """

    llm = create_agent_chat_llm(
        slot=resolve_consultation_model_slot(state),
        temperature=temperature,
        think=False,
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


def build_product_search_payload(
        *,
        final_result: ConsultationFinalDiagnosisSchema,
) -> dict[str, Any] | None:
    """
    功能描述：
        构造商品搜索工具入参。

    参数说明：
        final_result (ConsultationFinalDiagnosisSchema): 最终诊断结构化结果。

    返回值：
        dict[str, Any] | None: 搜索参数；无有效搜索线索时返回 `None`。

    异常说明：
        无。
    """

    payload = {
        "keyword": final_result.product_keyword,
        "usage": final_result.product_usage,
        "page_num": 1,
        "page_size": MAX_RECOMMENDED_PRODUCTS,
    }
    if not any([payload["keyword"], payload["usage"]]):
        return None
    return payload


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


def extract_candidate_products(search_result: Any) -> list[dict[str, Any]]:
    """
    功能描述：
        从商品搜索结果中提取候选商品列表。

    参数说明：
        search_result (Any): 商品搜索工具原始返回值。

    返回值：
        list[dict[str, Any]]: 归一化后的候选商品列表。

    异常说明：
        无。
    """

    if isinstance(search_result, Mapping):
        for key in ("rows", "items", "list"):
            raw_items = search_result.get(key)
            if isinstance(raw_items, list):
                return [dict(item) for item in raw_items if isinstance(item, Mapping)]

        raw_data = search_result.get("data")
        if isinstance(raw_data, Mapping):
            for key in ("rows", "items", "list"):
                raw_items = raw_data.get(key)
                if isinstance(raw_items, list):
                    return [dict(item) for item in raw_items if isinstance(item, Mapping)]

    return []


def extract_product_identity(product_item: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """
    功能描述：
        提取候选商品的商品 ID 与商品名称。

    参数说明：
        product_item (Mapping[str, Any]): 单个商品对象。

    返回值：
        tuple[int | None, str | None]:
            - 商品 ID；
            - 商品名称。

    异常说明：
        无。
    """

    raw_product_id = product_item.get("id")
    if raw_product_id is None:
        raw_product_id = product_item.get("productId")

    try:
        product_id = int(raw_product_id)
    except (TypeError, ValueError):
        product_id = None

    if product_id is not None and product_id <= 0:
        product_id = None

    raw_name = product_item.get("name")
    if raw_name is None:
        raw_name = product_item.get("productName")
    product_name = str(raw_name or "").strip() or None
    return product_id, product_name


def build_recommended_products(search_result: Any) -> tuple[list[int], list[str]]:
    """
    功能描述：
        从商品搜索结果中提取推荐商品 ID 与名称。

    参数说明：
        search_result (Any): 商品搜索工具原始返回值。

    返回值：
        tuple[list[int], list[str]]:
            - 推荐商品 ID 列表；
            - 推荐商品名称列表。

    异常说明：
        无。
    """

    product_ids: list[int] = []
    product_names: list[str] = []
    for item in extract_candidate_products(search_result):
        product_id, product_name = extract_product_identity(item)
        if product_id is None or product_id in product_ids:
            continue

        product_ids.append(product_id)
        if product_name:
            product_names.append(product_name)

        if len(product_ids) >= MAX_RECOMMENDED_PRODUCTS:
            break

    return product_ids, product_names


def build_recommendation_text(
        *,
        diagnosis_text: str,
        product_names: list[str],
) -> str:
    """
    功能描述：
        构造最终诊断节点返回给用户的文本。

    参数说明：
        diagnosis_text (str): 诊断文本。
        product_names (list[str]): 推荐药品名称列表。

    返回值：
        str: 最终展示文本。

    异常说明：
        无。
    """

    normalized_diagnosis = diagnosis_text.strip() or DEFAULT_DIAGNOSIS_TEXT
    if not product_names:
        return normalized_diagnosis

    medicine_text = "可考虑的药品有：" + "、".join(product_names) + "。"
    return build_text_result(normalized_diagnosis, medicine_text)


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


async def emit_consultation_answer_deltas_async(
        text: str,
        *,
        max_chunk_length: int = CONSULTATION_STREAM_CHUNK_MAX_LENGTH,
        emit_interval_seconds: float = CONSULTATION_STREAM_EMIT_INTERVAL_SECONDS,
) -> list[str]:
    """
    功能描述：
        以让渡事件循环的方式发送 consultation answer 增量事件。

    参数说明：
        text (str): 待发送的完整文本。
        max_chunk_length (int): 单个分片允许的最大长度。
        emit_interval_seconds (float): 每个分片之间的最小让渡间隔。

    返回值：
        list[str]: 实际发送的文本分片列表。

    异常说明：
        无。
    """

    stream_chunks = split_consultation_stream_text(
        text,
        max_chunk_length=max_chunk_length,
    )
    if not has_status_emitter():
        return stream_chunks

    for index, chunk_text in enumerate(stream_chunks):
        emit_answer_delta(chunk_text)
        if index < len(stream_chunks) - 1:
            await asyncio.sleep(max(emit_interval_seconds, 0))

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


def build_selection_card_response(
        *,
        title: str,
        options: list[str],
) -> AssistantResponse:
    """
    功能描述：
        构造 consultation 追问即时发送的选择卡片响应。

    参数说明：
        title (str): 卡片标题。
        options (list[str]): 卡片选项列表。

    返回值：
        AssistantResponse: 可直接推送给前端并持久化的卡片响应。

    异常说明：
        ValueError: 选择卡片参数不合法时由 schema 抛出。
    """

    request = SendSelectionCardRequest(
        title=title,
        options=options,
    )
    return build_card_response(
        Card(
            type=CONSULTATION_SELECTION_CARD_TYPE,
            data=request.to_card_data().model_dump(mode="json", exclude_none=True),
        ),
        persist_card=True,
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
    "CONSULTATION_INTERRUPT_KIND",
    "CONSULTATION_MODEL_SLOT_MAP",
    "CONSULTATION_SELECTION_CARD_TYPE",
    "CONSULTATION_STREAM_CHUNK_MAX_LENGTH",
    "CONSULTATION_STREAM_EMIT_INTERVAL_SECONDS",
    "CONSULTATION_STREAM_SENTENCE_ENDINGS",
    "CONSULTATION_STREAM_SOFT_BREAKS",
    "CONSULTATION_STREAM_SOFT_CHUNK_LENGTH",
    "DEFAULT_COMFORT_TEXT",
    "DEFAULT_DIAGNOSIS_TEXT",
    "DEFAULT_PURCHASE_QUANTITY",
    "DEFAULT_QUESTION_OPTIONS",
    "DEFAULT_QUESTION_REPLY_TEXT",
    "DEFAULT_QUESTION_TEXT",
    "DEFAULT_REPLY_TEXT",
    "DEFAULT_TEMPERATURE",
    "MAX_RECOMMENDED_PRODUCTS",
    "append_resume_messages",
    "append_trace_to_state",
    "build_consultation_graph_config",
    "build_interrupt_payload",
    "build_llm_agent",
    "build_product_search_payload",
    "build_recommendation_text",
    "build_recommended_products",
    "build_selection_card_response",
    "build_text_result",
    "build_tool_trace",
    "build_trace_item",
    "emit_consultation_answer_deltas_async",
    "extract_candidate_products",
    "extract_product_identity",
    "invoke_runnable",
    "merge_parallel_round_traces",
    "parse_json_text",
    "resolve_final_diagnosis_result",
    "resolve_interrupt_payload",
    "resolve_natural_language_text",
    "resolve_payload_text",
    "resolve_question_result",
    "resolve_resume_text",
    "run_async_safely",
    "split_consultation_stream_text",
]
