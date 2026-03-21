from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.agent.client.state import (
    ChatHistoryMessage,
    ExecutionTraceState,
    TokenUsageState,
)

# consultation 子图进行中状态。
CONSULTATION_STATUS_COLLECTING: Literal["collecting"] = "collecting"
# consultation 子图已完成最终诊断状态。
CONSULTATION_STATUS_COMPLETED: Literal["completed"] = "completed"
# consultation 子图允许的阶段值。
ConsultationStatusValue = Literal["collecting", "completed"]


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


class ConsultationState(TypedDict, total=False):
    """Client 病情咨询子图状态。"""

    # consultation 子图当前可见的完整历史消息，包含用户消息与已写回的 AI 回复。
    history_messages: list[ChatHistoryMessage]
    # 当前会话的任务难度，用于决定 consultation 子图的处理策略。
    task_difficulty: str
    # consultation 子图当前所处阶段，区分仍在收集信息还是已完成诊断。
    consultation_status: ConsultationStatusValue
    # 当前信息是否已经足够直接进入最终诊断节点。
    diagnosis_ready: bool
    # 安抚节点产出的流式安抚文本。
    comfort_text: str
    # 追问节点产出的阶段性分析文本。
    question_reply_text: str
    # 当前轮待展示给用户的追问标题。
    pending_question_text: str
    # 当前轮待展示给用户的追问候选选项。
    pending_question_options: list[str]
    # 当前轮真正展示给用户的完整 AI 回复文本，用于 interrupt 恢复后回写历史。
    pending_ai_reply_text: str
    # 最终诊断节点产出的完整诊断文本。
    final_text: str
    # consultation 子图累计的节点执行轨迹列表。
    execution_traces: list[ExecutionTraceState]
    # consultation 子图当前聚合后的 token 使用量。
    token_usage: TokenUsageState | None
    # 安抚节点的单节点执行轨迹。
    comfort_trace: ExecutionTraceState
    # 追问节点的单节点执行轨迹。
    question_trace: ExecutionTraceState
    # 最终诊断节点的单节点执行轨迹。
    diagnosis_trace: ExecutionTraceState
    # interrupt 节点的单节点执行轨迹。
    interrupt_trace: ExecutionTraceState
    # 当前待恢复的 interrupt 负载；无待恢复中断时为空。
    interrupt_payload: ConsultationInterruptPayload | None
    # 最近一次 resume 时用户补充的原始文本。
    last_resume_text: str
    # consultation 子图当前轮最终结果文本，供父图读取。
    result: str
    # consultation 子图当前轮需要回写给 LangGraph 的消息列表。
    messages: list[Any]


__all__ = [
    "CONSULTATION_STATUS_COLLECTING",
    "CONSULTATION_STATUS_COMPLETED",
    "ConsultationInterruptPayload",
    "ConsultationState",
    "ConsultationStatusValue",
]
