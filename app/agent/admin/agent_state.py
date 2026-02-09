from typing import TypedDict, List, Dict, Any, Optional, Union


class OrderContext(TypedDict, total=False):
    """
    订单 Agent 使用的上下文
    """

    # 需要处理的订单 ID 列表
    order_ids: List[str]

    # 订单查询条件
    order_query_conditions: Dict[str, Any]

    # 查询得到的订单数据
    order_data: List[Dict[str, Any]]

    # 订单 Agent 当前状态
    status: str


class AfterSaleContext(TypedDict, total=False):
    """
    售后 Agent 使用的上下文
    """
    # 售后或退款单 ID 列表
    refund_ids: List[str]

    # 售后类型
    aftersale_type: str

    # 售后处理结果数据
    aftersale_data: List[Dict[str, Any]]

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
    columns: List[str]

    # Excel 文件路径
    file_path: Optional[str]

    # Excel Agent 当前状态
    status: str


class PlanStep(TypedDict, total=False):
    """
    单个执行步骤
    """

    # 执行节点名称
    node_name: str

    # 上层节点名称
    last_node:list[str]

    # 任务描述
    task_description: str


class AgentState(TypedDict):
    """
    LangGraph 全局状态
    """

    # 用户原始输入
    user_input: str

    # 解析后的用户意图
    user_intent: Dict[str, Any]

    # 执行计划
    plan: List[Union[PlanStep, List[PlanStep]]]

    # 当前执行状态
    routing: Dict[str, Any]

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
