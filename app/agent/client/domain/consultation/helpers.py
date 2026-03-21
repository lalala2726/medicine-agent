from __future__ import annotations

import asyncio
import contextvars
import json
from collections.abc import Mapping
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from pydantic import ValidationError

from app.agent.client.domain.consultation.schema import (
    ConsultationFinalDiagnosisSchema,
    ConsultationQuestionSchema,
    ConsultationStatusSchema,
)
from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    CONSULTATION_STATUS_COMPLETED,
    ConsultationState,
)
from app.agent.client.state import ExecutionTraceState, ToolCallTraceState
from app.core.agent.agent_event_bus import emit_answer_delta, has_status_emitter
from app.core.agent.agent_tool_trace import extract_text
from app.core.agent.base_prompt_middleware import BasePromptMiddleware
from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.utils.prompt_utils import append_current_time_to_prompt

# consultation 节点兜底的提问文本。
DEFAULT_QUESTION_TEXT = "为了更准确判断你的情况，我还需要再确认几个关键信息。"
# consultation 节点兜底的选项卡列表。
DEFAULT_QUESTION_OPTIONS = ["有发热", "有咳痰", "症状超过3天", "都没有这些情况"]
# consultation 节点兜底的安抚文本。
DEFAULT_COMFORT_TEXT = (
    "我先帮你做一个常见轻症方向的初步判断，但这不能替代医生面诊。"
    "如果你有高热不退、呼吸困难、胸痛、持续加重等情况，请尽快线下就医。"
)
# consultation 最终诊断节点兜底文本。
DEFAULT_DIAGNOSIS_TEXT = "结合你现在提供的信息，我建议先观察症状变化；如果持续加重，请及时线下就医。"
# consultation 最终推荐商品数量上限。
MAX_RECOMMENDED_PRODUCTS = 3
# consultation 购买确认卡默认数量。
DEFAULT_PURCHASE_QUANTITY = 1
# consultation 节点默认温度配置。
DEFAULT_TEMPERATURE = 0.2
# consultation 统一出口流式时单个文本分片的最大长度。
CONSULTATION_STREAM_CHUNK_MAX_LENGTH = 80
# consultation 统一出口流式优先切句的标点集合。
CONSULTATION_STREAM_SENTENCE_ENDINGS = frozenset({"。", "！", "？", "；"})
# consultation 节点允许的 task_difficulty 到模型槽位映射。
CONSULTATION_MODEL_SLOT_MAP: dict[str, AgentChatModelSlot] = {
    "normal": AgentChatModelSlot.BUSINESS_SIMPLE,
    "high": AgentChatModelSlot.BUSINESS_COMPLEX,
}


def run_async_safely(coro: Any) -> Any:
    """在同步节点中安全执行异步协程。"""

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
    """执行工具对象，优先走异步调用。"""

    ainvoke = getattr(tool_object, "ainvoke", None)
    if callable(ainvoke):
        return run_async_safely(ainvoke(payload))
    invoke = getattr(tool_object, "invoke", None)
    if callable(invoke):
        return invoke(payload)
    raise TypeError(f"tool_object {tool_object!r} does not support invoke/ainvoke")


def resolve_consultation_model_slot(state: ConsultationState) -> AgentChatModelSlot:
    """根据 consultation 的 task_difficulty 选择模型槽位。"""

    task_difficulty = str(state.get("task_difficulty") or "").strip().lower()
    return CONSULTATION_MODEL_SLOT_MAP.get(
        task_difficulty,
        AgentChatModelSlot.BUSINESS_SIMPLE,
    )


def resolve_payload_text(raw_payload: Any) -> str:
    """从 agent payload 中提取最后一条 AI 文本。"""

    if isinstance(raw_payload, Mapping):
        raw_messages = raw_payload.get("messages")
        if isinstance(raw_messages, list) and raw_messages:
            last_message = raw_messages[-1]
            text = extract_text(last_message).strip()
            if text:
                return text
            raw_content = getattr(last_message, "content", None)
            if isinstance(raw_content, str):
                return raw_content.strip()
    return ""


def parse_json_text(raw_payload: Any) -> dict[str, Any] | None:
    """解析节点输出中的 JSON 文本。"""

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


def resolve_status_result(raw_payload: Any) -> ConsultationStatusSchema:
    """解析 consultation 状态判断输出。"""

    parsed_json = parse_json_text(raw_payload)
    if parsed_json is None:
        return ConsultationStatusSchema(
            should_enter_diagnosis=False,
            consultation_status=CONSULTATION_STATUS_COLLECTING,
        )
    try:
        return ConsultationStatusSchema.model_validate(parsed_json)
    except ValidationError:
        return ConsultationStatusSchema(
            should_enter_diagnosis=False,
            consultation_status=CONSULTATION_STATUS_COLLECTING,
        )


def resolve_question_result(raw_payload: Any) -> ConsultationQuestionSchema:
    """解析 consultation 问询卡片输出。"""

    parsed_json = parse_json_text(raw_payload)
    if parsed_json is None:
        return ConsultationQuestionSchema(
            should_enter_diagnosis=False,
            consultation_status=CONSULTATION_STATUS_COLLECTING,
            question_text=DEFAULT_QUESTION_TEXT,
            options=list(DEFAULT_QUESTION_OPTIONS),
        )
    try:
        return ConsultationQuestionSchema.model_validate(parsed_json)
    except ValidationError:
        return ConsultationQuestionSchema(
            should_enter_diagnosis=False,
            consultation_status=CONSULTATION_STATUS_COLLECTING,
            question_text=DEFAULT_QUESTION_TEXT,
            options=list(DEFAULT_QUESTION_OPTIONS),
        )


def resolve_final_result(raw_payload: Any) -> ConsultationFinalDiagnosisSchema:
    """解析 consultation 最终诊断输出。"""

    parsed_json = parse_json_text(raw_payload)
    if parsed_json is None:
        return ConsultationFinalDiagnosisSchema(
            diagnosis_text=DEFAULT_DIAGNOSIS_TEXT,
            should_recommend_products=False,
        )
    try:
        return ConsultationFinalDiagnosisSchema.model_validate(parsed_json)
    except ValidationError:
        return ConsultationFinalDiagnosisSchema(
            diagnosis_text=DEFAULT_DIAGNOSIS_TEXT,
            should_recommend_products=False,
        )


def build_tool_trace(
        *,
        tool_name: str,
        tool_input: dict[str, Any],
) -> ToolCallTraceState:
    """构造手工工具调用追踪结构。"""

    return ToolCallTraceState(
        tool_name=tool_name,
        tool_call_id=None,
        tool_input=tool_input,
    )


def build_trace_item(
        *,
        node_name: str,
        llm_model_name: str,
        trace_payload: dict[str, Any],
        tool_calls: list[ToolCallTraceState] | None = None,
        node_context: dict[str, Any] | None = None,
) -> ExecutionTraceState:
    """构造 consultation 节点 execution trace。"""

    trace_model_name = str(trace_payload.get("model_name") or "").strip()
    return ExecutionTraceState(
        sequence=0,
        node_name=node_name,
        model_name=llm_model_name or trace_model_name or "unknown",
        status="success",
        output_text=str(trace_payload.get("text") or "").strip(),
        llm_usage_complete=bool(trace_payload.get("is_usage_complete", False)),
        llm_token_usage=trace_payload.get("usage"),
        tool_calls=list(tool_calls or []),
        node_context=node_context,
    )


def build_llm_agent(
        *,
        state: ConsultationState,
        prompt_text: str,
        temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[Any, str]:
    """构造 consultation 节点使用的 agent。"""

    llm = create_agent_chat_llm(
        slot=resolve_consultation_model_slot(state),
        temperature=temperature,
        think=False,
    )
    llm_model_name = str(getattr(llm, "model_name", "") or "").strip()
    agent = create_agent(
        model=llm,
        system_prompt=SystemMessage(
            content=append_current_time_to_prompt(prompt_text)
        ),
        middleware=[
            BasePromptMiddleware(base_prompt_file="client/_client_base_prompt.md"),
        ],
    )
    return agent, llm_model_name


def build_text_result(*parts: str) -> str:
    """按固定顺序拼接最终回复文本。"""

    normalized_parts = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    return "\n\n".join(normalized_parts)


def build_product_search_payload(
        *,
        final_result: ConsultationFinalDiagnosisSchema,
) -> dict[str, Any] | None:
    """构造商品搜索工具入参。"""

    payload: dict[str, Any] = {
        "keyword": final_result.product_keyword,
        "usage": final_result.product_usage,
        "page_num": 1,
        "page_size": MAX_RECOMMENDED_PRODUCTS,
    }
    if not any([payload["keyword"], payload["usage"]]):
        return None
    return payload


def extract_candidate_products(search_result: Any) -> list[dict[str, Any]]:
    """从商品搜索结果中提取候选商品列表。"""

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
    """提取候选商品的商品 ID 与商品名称。"""

    raw_product_id = product_item.get("id")
    if raw_product_id is None:
        raw_product_id = product_item.get("productId")
    product_id: int | None
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
    """从商品搜索结果中提取推荐商品 ID 与名称。"""

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
    """构造最终诊断节点返回给用户的文本。"""

    normalized_diagnosis = diagnosis_text.strip() or DEFAULT_DIAGNOSIS_TEXT
    if not product_names:
        return normalized_diagnosis
    medicine_text = "可考虑的药品有：" + "、".join(product_names) + "。"
    return build_text_result(normalized_diagnosis, medicine_text)


def resolve_stream_response_text(state: ConsultationState) -> str:
    """解析 consultation 统一出口节点应输出的最终文本。"""

    final_text = str(state.get("final_text") or "").strip()
    if final_text:
        return final_text

    consultation_status = str(state.get("consultation_status") or "").strip().lower()
    if consultation_status == CONSULTATION_STATUS_COMPLETED:
        return DEFAULT_DIAGNOSIS_TEXT

    collecting_text = build_text_result(
        str(state.get("comfort_text") or ""),
        str(state.get("question_text") or ""),
    )
    if collecting_text:
        return collecting_text

    return build_text_result(DEFAULT_COMFORT_TEXT, DEFAULT_QUESTION_TEXT)


def _hard_split_stream_chunk(
        text: str,
        *,
        max_chunk_length: int,
) -> list[str]:
    """按固定长度对超长文本做硬切分，且保持原文不改写。"""

    normalized_text = str(text or "")
    if not normalized_text:
        return []
    return [
        normalized_text[index:index + max_chunk_length]
        for index in range(0, len(normalized_text), max_chunk_length)
    ]


def _split_stream_paragraph(
        paragraph_text: str,
        *,
        max_chunk_length: int,
) -> list[str]:
    """优先按句末标点拆分单个段落，再对超长句子做硬切分。"""

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
        if len(sentence) <= max_chunk_length:
            stream_chunks.append(sentence)
            continue
        stream_chunks.extend(
            _hard_split_stream_chunk(
                sentence,
                max_chunk_length=max_chunk_length,
            )
        )
    return stream_chunks


def split_consultation_stream_text(
        text: str,
        *,
        max_chunk_length: int = CONSULTATION_STREAM_CHUNK_MAX_LENGTH,
) -> list[str]:
    """按 consultation 统一出口规则切分流式文本分片。"""

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


def emit_consultation_answer_deltas(
        text: str,
        *,
        max_chunk_length: int = CONSULTATION_STREAM_CHUNK_MAX_LENGTH,
) -> list[str]:
    """按 consultation 统一出口规则发送 answer 增量事件。"""

    stream_chunks = split_consultation_stream_text(
        text,
        max_chunk_length=max_chunk_length,
    )
    if not has_status_emitter():
        return stream_chunks

    for chunk_text in stream_chunks:
        emit_answer_delta(chunk_text)
    return stream_chunks


def collect_consultation_traces(state: ConsultationState) -> list[ExecutionTraceState]:
    """按稳定顺序提取 consultation 子图 trace 列表。"""

    traces: list[ExecutionTraceState] = []
    for key in (
            "status_trace",
            "comfort_trace",
            "question_trace",
            "diagnosis_trace",
    ):
        raw_trace = state.get(key)
        if not isinstance(raw_trace, Mapping):
            continue
        node_name = str(raw_trace.get("node_name") or "").strip()
        if not node_name:
            continue
        traces.append(ExecutionTraceState(**dict(raw_trace)))
    return traces


def resequence_traces(
        *,
        existing_traces: list[ExecutionTraceState],
        appended_traces: list[ExecutionTraceState],
) -> list[ExecutionTraceState]:
    """按父图已有 trace 长度重排序 consultation trace。"""

    resequenced_traces = list(existing_traces)
    base_sequence = len(existing_traces)
    for offset, raw_trace in enumerate(appended_traces, start=1):
        trace_payload = dict(raw_trace)
        trace_payload["sequence"] = base_sequence + offset
        resequenced_traces.append(ExecutionTraceState(**trace_payload))
    return resequenced_traces


__all__ = [
    "CONSULTATION_STREAM_CHUNK_MAX_LENGTH",
    "CONSULTATION_STREAM_SENTENCE_ENDINGS",
    "CONSULTATION_MODEL_SLOT_MAP",
    "DEFAULT_COMFORT_TEXT",
    "DEFAULT_DIAGNOSIS_TEXT",
    "DEFAULT_PURCHASE_QUANTITY",
    "DEFAULT_QUESTION_OPTIONS",
    "DEFAULT_QUESTION_TEXT",
    "DEFAULT_TEMPERATURE",
    "MAX_RECOMMENDED_PRODUCTS",
    "build_llm_agent",
    "build_product_search_payload",
    "build_recommendation_text",
    "build_recommended_products",
    "build_text_result",
    "build_tool_trace",
    "build_trace_item",
    "collect_consultation_traces",
    "emit_consultation_answer_deltas",
    "extract_candidate_products",
    "extract_product_identity",
    "invoke_runnable",
    "parse_json_text",
    "resequence_traces",
    "resolve_stream_response_text",
    "resolve_consultation_model_slot",
    "resolve_final_result",
    "resolve_payload_text",
    "resolve_question_result",
    "resolve_status_result",
    "run_async_safely",
    "split_consultation_stream_text",
]
