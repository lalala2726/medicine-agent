from typing import TypedDict, List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field


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


class PlanStep(TypedDict, total=False):
    """
    单个执行步骤
    """

    # 执行节点名称
    node_name: str

    # 上层节点名称
    last_node: list[str]

    # 任务描述
    task_description: str


class RoutingState(TypedDict, total=False):
    """
    路由状态（供 workflow/router 与各 agent 共享）
    """

    # 当前执行到第几个阶段（并行编排使用）
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


class AgentState(TypedDict):
    """
    LangGraph 全局状态
    """

    # 用户原始输入
    user_input: str

    # 解析后的用户意图
    user_intent: Dict[str, Any]

    # 执行计划
    plan: list[Union[PlanStep, list[PlanStep]]]

    # 当前执行状态
    routing: RoutingState

    # 订单 Agent 上下文
    order_context: OrderContext

    # 售后 Agent 上下文
    aftersale_context: AfterSaleContext

    # Excel Agent 上下文
    excel_context: ExcelContext

    # 共享信息
    shared_memory: Dict[str, Any]

    # 最终结果
    results: Dict[str, Any]

    # 错误信息
    errors: List[str]
