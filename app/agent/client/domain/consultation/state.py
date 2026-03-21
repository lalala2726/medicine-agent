from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypedDict

from app.agent.client.state import (
    ChatHistoryMessage,
    ExecutionTraceState,
    TokenUsageState,
)

# consultation 子图进行中状态。
CONSULTATION_STATUS_COLLECTING: Literal["collecting"] = "collecting"
# consultation 子图已完成状态。
CONSULTATION_STATUS_COMPLETED: Literal["completed"] = "completed"
# consultation 子图允许的阶段值。
ConsultationStatusValue = Literal["collecting", "completed"]
# consultation 子图允许的路由动作。
ConsultationNextActionValue = Literal["reply_only", "ask_followup", "final_diagnosis"]
# consultation 子图允许的咨询模式。
ConsultationModeValue = Literal["simple_medical", "diagnostic_consultation"]


class ConsultationInterruptPayload(TypedDict, total=False):
    """consultation 追问中断负载。"""

    # 中断负载类型标识，用于区分不同 interrupt 来源。
    kind: str
    # 追问卡片中展示给用户的阶段性分析文本。
    reply_text: str
    # 追问卡片标题使用的核心问题文本。
    question_text: str
    # 追问卡片可点击的候选选项列表。
    options: list[str]


class ConsultationRouteState(TypedDict, total=False):
    """consultation 子图内路由结果。"""

    # 本轮 consultation 应执行的下一跳动作。
    next_action: ConsultationNextActionValue
    # 当前问题属于简单医学询问还是诊断型咨询。
    consultation_mode: ConsultationModeValue
    # 路由节点给出的简要原因，便于 trace 与排查。
    reason: str


class ConsultationFollowupRecordState(TypedDict, total=False):
    """consultation 已追问记录。"""

    # 本次追问对应的结构化槽位标识。
    slot_key: str
    # 本次追问展示给用户的问题标题。
    question_text: str
    # 本次追问展示给用户的选项列表。
    options: list[str]
    # 用户对该追问的最终回答文本。
    answer_text: str


class ConsultationProgressState(TypedDict, total=False):
    """consultation 追问进度状态。"""

    # 历史上已经发给用户并完成回答的追问记录。
    asked_followups: list[ConsultationFollowupRecordState]
    # 历史上已经问过的槽位标识集合。
    asked_slots: list[str]
    # 历史上已经得到回答的槽位与答案映射。
    answered_slots: dict[str, str]
    # 当前轮待恢复追问所对应的槽位标识。
    pending_slot_key: str


class ConsultationResponseOutputState(TypedDict, total=False):
    """consultation 医学回应节点用户可见输出。"""

    # 医学回应节点流式输出给用户的正文文本。
    text: str


class ConsultationQuestionOutputState(TypedDict, total=False):
    """consultation 追问节点用户可见输出。"""

    # 追问阶段展示给用户的阶段性分析文本。
    reply_text: str
    # 追问卡片标题使用的核心问题文本。
    question_text: str
    # 追问卡片可点击的候选选项列表。
    options: list[str]
    # 当前轮真正展示给用户的完整 AI 回复文本，用于 interrupt 恢复后回写历史。
    ai_reply_text: str


class ConsultationFinalDiagnosisOutputState(TypedDict, total=False):
    """consultation 最终诊断节点用户可见输出。"""

    # 最终诊断节点输出给用户的完整诊断正文。
    text: str


class ConsultationInterruptOutputState(TypedDict, total=False):
    """consultation interrupt 阶段用户可见输出。"""

    # 当前轮待恢复的 interrupt 负载。
    payload: ConsultationInterruptPayload | None


class ConsultationOutputsState(TypedDict, total=False):
    """consultation 子图统一用户可见输出容器。"""

    # 医学回应节点用户可见输出。
    response: ConsultationResponseOutputState
    # 追问节点用户可见输出。
    question: ConsultationQuestionOutputState
    # 最终诊断节点用户可见输出。
    final_diagnosis: ConsultationFinalDiagnosisOutputState
    # interrupt 阶段用户可见输出。
    interrupt: ConsultationInterruptOutputState


def merge_consultation_outputs(
        left_value: ConsultationOutputsState | Mapping[str, Any] | None,
        right_value: ConsultationOutputsState | Mapping[str, Any] | None,
) -> ConsultationOutputsState:
    """
    功能描述：
        合并 consultation 并行节点写回的统一输出容器，避免 LangGraph 在同一步
        写入 `consultation_outputs` 时触发 LastValue 冲突。

    参数说明：
        left_value (ConsultationOutputsState | Mapping[str, Any] | None): 先前已存在的输出容器。
        right_value (ConsultationOutputsState | Mapping[str, Any] | None): 当前节点新写回的输出容器。

    返回值：
        ConsultationOutputsState: 逐层合并后的输出容器。

    异常说明：
        无。
    """

    merged_outputs: dict[str, Any] = {}

    for candidate in (left_value, right_value):
        if not isinstance(candidate, Mapping):
            continue
        for section_name, section_value in candidate.items():
            if not isinstance(section_value, Mapping):
                merged_outputs[section_name] = section_value
                continue

            current_section = dict(merged_outputs.get(section_name) or {})
            current_section.update(dict(section_value))
            merged_outputs[section_name] = current_section

    return ConsultationOutputsState(**merged_outputs)


class ConsultationState(TypedDict, total=False):
    """Client 病情咨询子图状态。"""

    # consultation 子图当前可见的完整历史消息，包含用户消息与已写回的 AI 回复。
    history_messages: list[ChatHistoryMessage]
    # 当前会话的任务难度。
    task_difficulty: str
    # consultation 子图当前所处阶段。
    consultation_status: ConsultationStatusValue
    # 当前信息是否已经足够直接进入最终诊断节点。
    diagnosis_ready: bool
    # consultation 子图内路由节点的结构化决策结果。
    consultation_route: ConsultationRouteState
    # consultation 子图内追问去重与回答进度状态。
    consultation_progress: ConsultationProgressState
    # consultation 子图统一的用户可见输出容器。
    consultation_outputs: Annotated[ConsultationOutputsState, merge_consultation_outputs]
    # consultation 子图累计的节点执行轨迹列表。
    execution_traces: list[ExecutionTraceState]
    # consultation 子图当前聚合后的 token 使用量。
    token_usage: TokenUsageState | None
    # 医学回应节点的单节点执行轨迹。
    response_trace: ExecutionTraceState
    # 追问节点的单节点执行轨迹。
    question_trace: ExecutionTraceState
    # 路由节点的单节点执行轨迹。
    route_trace: ExecutionTraceState
    # 最终诊断节点的单节点执行轨迹。
    diagnosis_trace: ExecutionTraceState
    # interrupt 节点的单节点执行轨迹。
    interrupt_trace: ExecutionTraceState
    # 最近一次 resume 时用户补充的原始文本。
    last_resume_text: str
    # consultation 子图当前轮最终结果文本，供父图读取。
    result: str
    # consultation 子图当前轮需要回写给 LangGraph 的消息列表。
    messages: list[Any]


__all__ = [
    "CONSULTATION_STATUS_COLLECTING",
    "CONSULTATION_STATUS_COMPLETED",
    "ConsultationFinalDiagnosisOutputState",
    "ConsultationFollowupRecordState",
    "ConsultationInterruptOutputState",
    "ConsultationInterruptPayload",
    "ConsultationModeValue",
    "ConsultationNextActionValue",
    "ConsultationOutputsState",
    "ConsultationProgressState",
    "ConsultationQuestionOutputState",
    "ConsultationResponseOutputState",
    "ConsultationRouteState",
    "ConsultationState",
    "ConsultationStatusValue",
    "merge_consultation_outputs",
]
