from langchain_core.prompts import SystemMessagePromptTemplate

from app.agent.admin.agent_utils import invoke_with_failure_policy
from app.agent.admin.agent_state import AgentState
from app.agent.admin.node.runtime_context import (
    build_instruction_with_failure_policy,
    build_step_output_update,
    build_step_runtime,
    evaluate_failure_by_policy,
)
from app.agent.tools.admin_tools import get_orders_detail, get_order_list
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import is_final_node

system_prompt = (
        """
        # 系统角色定义
        你是运行在智能药品商城系统中的一个智能体节点。系统中同时存在多个智能体节点，
        你必须严格服从协调节点的统一调度与指挥，请你按照指令行事。
    
        # 职责范围
        你的唯一职责是处理与订单相关的业务，包括但不限于：
        - 订单查询
        - 订单校验
        - 订单状态判断
    
        # 订单有效性约束
        你只能处理真实且有效的订单数据：
        - 若订单信息不存在、缺失或无法校验为有效订单，必须返回空结果
        - 严禁构造、猜测、补全或返回任何虚假的订单信息
        你在获取订单列表的时候就能获取大部分订单信息，除非用户让你查询订单的详细信息，你才主动去调用获取订单详细信息的工具。
        你每次只能调用一个工具，并且必须拿到订单数据之后才能进行下一步
    
        # 行为准则
        - 不得处理非订单业务
        - 不得返回虚假数据
        - 不得违背协调节点的调度指令
    
        下面是你的任务描述: {instruction}
         """
        + base_prompt
)


@status_node(
    node="order",
    start_message="正在处理订单问题",
    display_when="after_coordinator",
)
@traceable(name="Order Agent Node", run_type="chain")
def order_agent(state: AgentState) -> dict:
    # 统一读取当前步骤运行时信息：
    # - task_description
    # - read_from 对应的上游输出
    # - include_user_input / include_chat_history 开关
    runtime = build_step_runtime(
        state,
        "order_agent",
        default_task_description="请处理订单相关任务",
    )
    # 生成给模型的 instruction 文本（统一包含失败策略提示拼接逻辑）。
    instruction = build_instruction_with_failure_policy(runtime)
    failure_policy = runtime.get("failure_policy") or {}

    llm = create_chat_model(model="qwen3-max")
    messages = SystemMessagePromptTemplate.from_template(
        template=system_prompt
    ).format_messages(instruction=instruction)

    tools = [get_orders_detail, get_order_list]
    # final_output=true 时允许节点内部 stream（用于最终对用户输出的节点）。
    final_output = is_final_node(state, "order_agent")
    try:
        content, diagnostics = invoke_with_failure_policy(
            llm=llm,
            messages=messages,
            tools=tools,
            enable_stream=final_output,
            failure_policy=failure_policy,
        )
        stream_chunks = list(diagnostics.get("stream_chunks") or [])
        step_status, failed_error, content = evaluate_failure_by_policy(
            content,
            diagnostics,
            failure_policy,
        )
    except Exception as exc:
        content = "订单服务暂时不可用，请稍后重试。"
        stream_chunks = []
        step_status = "failed"
        failed_error = f"order_agent 执行失败: {exc}"

    # 保留原有 order_context 结构，避免影响现有调用方。
    order_context = dict(state.get("order_context") or {})
    order_context["result"] = {
        "content": content,
        "is_end": final_output,
    }
    if stream_chunks:
        order_context["stream_chunks"] = stream_chunks
    else:
        order_context.pop("stream_chunks", None)
    order_context["status"] = "COMPLETED" if step_status == "completed" else "FAILED"

    # 额外写入标准化 step_outputs，供 planner 下一轮调度使用。
    result = {"order_context": order_context}
    result.update(
        build_step_output_update(
            runtime,
            node_name="order_agent",
            status=step_status,
            text=content,
            output={"result": order_context["result"]},
            error=failed_error,
        )
    )
    return result
