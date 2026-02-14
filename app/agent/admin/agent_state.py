from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, TypedDict


class HistoryMessage(TypedDict):
    """
    对话历史消息。
    """

    role: Literal["user", "assistant"]
    content: str


class StepOutput(TypedDict, total=False):
    """
    DAG 步骤执行输出。
    """

    step_id: str
    node_name: str
    status: Literal["completed", "failed", "skipped"]
    text: str
    output: Dict[str, Any]
    error: str


class FallbackFailedStep(TypedDict, total=False):
    """
    fallback 场景下的失败步骤信息。
    """

    step_id: str
    node_name: str
    status: Literal["failed", "skipped"]
    error: str


class FallbackPartialResult(TypedDict, total=False):
    """
    fallback 场景下可用于对用户展示的部分成功结果。
    """

    step_id: str
    node_name: str
    text: str


class FallbackContext(TypedDict, total=False):
    """
    planner 触发兜底 chat 时写入的上下文。
    """

    trigger: Literal["final_output_unreachable"]
    final_step_id: str
    failed_steps: List[FallbackFailedStep]
    partial_results: List[FallbackPartialResult]
    reason_text: str


def _merge_step_outputs(
        left: dict[str, StepOutput] | None,
        right: dict[str, StepOutput] | None,
) -> dict[str, StepOutput]:
    """
    合并并行节点写入的 step_outputs。
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class OrderContext(TypedDict, total=False):
    """
    订单 Agent 使用的上下文
    """

    # 结果
    result: Dict[str, Any]

    # 订单 Agent 当前状态
    status: str


class AfterSaleContext(TypedDict, total=False):
    """
    售后 Agent 使用的上下文
    """
    # 售后或退款单 ID 列表
    refund_ids: list[str]

    # 售后类型
    aftersale_type: str

    # 售后处理结果数据
    aftersale_data: list[Dict[str, Any]]

    # 售后 Agent 当前状态
    status: str


class ExcelContext(TypedDict, total=False):
    """
    Excel Agent 使用的上下文
    """
    # 导出类型
    export_type: str

    # 数据来源 Agent
    source_agent: str

    # 导出的列名
    columns: list[str]

    # URL 地址
    url: str

    # Excel Agent 当前状态
    status: str


class ProductContext(TypedDict, total=False):
    """
    商品 Agent 使用的上下文
    """
    # 结果
    result: Dict[str, Any]

    # 商品 Agent 当前状态
    status: str


class StepFailurePolicy(TypedDict, total=False):
    """
    单步骤失败判定策略。
    """

    # 失败判定模式：
    # - hybrid: 哨兵与工具阈值任一命中即失败
    # - marker_only: 仅哨兵命中失败
    # - tool_only: 仅工具阈值命中失败
    mode: Literal["hybrid", "marker_only", "tool_only"]

    # 模型错误哨兵前缀（命中即判定失败）
    error_marker_prefix: str

    # 工具失败计数方式：
    # - consecutive: 连续失败计数
    # - total: 累计失败计数
    tool_error_counting: Literal["consecutive", "total"]

    # 工具失败阈值，达到即判失败（范围由规则层校验）
    max_tool_errors: int

    # 是否启用严格数据质量要求（用于提示词约束）
    strict_data_quality: bool


class PlanStep(TypedDict, total=False):
    """
    单个执行步骤
    """

    # 步骤唯一 ID
    step_id: str

    # 执行节点名称
    node_name: str

    # 必选依赖步骤 ID（必须 completed）
    required_depends_on: list[str]

    # 可选依赖步骤 ID（允许 failed/skipped，但需先进入终态）
    optional_depends_on: list[str]

    # 读取的上游步骤 ID
    read_from: list[str]

    # 任务描述
    task_description: str

    # 是否允许读取 user_input
    include_user_input: bool

    # 是否允许读取 history_messages
    include_chat_history: bool

    # 是否为最终输出步骤（全局必须唯一）
    final_output: bool

    # 步骤级失败策略（可选，缺省由系统默认值补齐）
    failure_policy: StepFailurePolicy


class RoutingState(TypedDict, total=False):
    """
    路由状态（供 workflow/router 与各 agent 共享）
    """

    # 当前执行到第几个阶段（兼容旧字段）
    stage_index: int

    # router 计算出的下一批可执行节点（单节点或并行多节点）
    next_nodes: list[str]

    # gateway_router 的路由结果
    route_target: str

    # 任务难度（simple / medium / complex）
    difficulty: str

    # 当前阶段是否为最后一个有效执行阶段
    is_final_stage: bool

    # 当前阶段节点到步骤定义的映射
    current_step_map: Dict[str, PlanStep]

    # 当前调度批次步骤 ID 列表
    current_step_ids: list[str]

    # 下一批可执行步骤 ID 列表
    next_step_ids: list[str]

    # 已完成步骤 ID 列表
    completed_step_ids: list[str]

    # 被阻断/跳过步骤 ID 列表
    blocked_step_ids: list[str]

    # fallback 到 chat 节点时的上下文
    fallback_context: FallbackContext


class AgentState(TypedDict):
    """
    LangGraph 全局状态
    """

    # 用户原始输入
    user_input: str

    # 解析后的用户意图
    user_intent: Dict[str, Any]

    # 执行计划
    plan: list[PlanStep]

    # 当前执行状态
    routing: RoutingState

    # 订单 Agent 上下文
    order_context: OrderContext

    # 售后 Agent 上下文
    aftersale_context: AfterSaleContext

    # Excel Agent 上下文
    excel_context: ExcelContext

    # 商品 Agent 上下文
    product_context: ProductContext

    # 历史消息（用户与助手）
    history_messages: List[HistoryMessage]

    # DAG 步骤产出（并发安全聚合）
    step_outputs: Annotated[dict[str, StepOutput], _merge_step_outputs]

    # 共享信息
    shared_memory: Dict[str, Any]

    # 最终结果
    results: Dict[str, Any]

    # 错误信息
    errors: List[str]
