from langchain_core.prompts import SystemMessagePromptTemplate

from app.agent.admin.agent_state import AgentState
from app.agent.tools.admin_tools import get_orders_detail, get_order_list
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke_with_optional_stream, is_final_node

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
    routing = state.get("routing") or {}
    current_step_map = routing.get("current_step_map") or {}
    step = current_step_map.get("order_agent")
    instruction = (
            step.get("task_description") if isinstance(step, dict) else None
    ) or state.get("user_input") or "请处理订单相关任务"

    llm = create_chat_model(model="qwen3-max")
    messages = SystemMessagePromptTemplate.from_template(
        template=system_prompt
    ).format_messages(instruction=instruction)

    tools = [get_orders_detail, get_order_list]

    final_output = is_final_node(state, "order_agent")
    content, stream_chunks = invoke_with_optional_stream(
        llm,
        messages,
        tools=tools,
        enable_stream=final_output,
    )

    order_context = dict(state.get("order_context") or {})
    order_context["result"] = {
        "content": content,
        "is_end": final_output,
    }
    if stream_chunks:
        order_context["stream_chunks"] = stream_chunks
    else:
        order_context.pop("stream_chunks", None)
    order_context["status"] = "COMPLETED"

    return {"order_context": order_context}
