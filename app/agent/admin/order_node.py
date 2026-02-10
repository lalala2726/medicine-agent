from langchain_core.prompts import SystemMessagePromptTemplate

from app.agent.admin.agent_state import AgentState
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from schemas.prompt import base_prompt

system_prompt = """
    # 系统角色定义
    你是运行在智能药品商城系统中的一个智能体节点。系统中同时存在多个智能体节点，
    你必须严格服从协调节点的统一调度与指挥，请你按照指令行事。
    
    # 职责范围
    你的唯一职责是处理与订单相关的业务，包括但不限于：
    - 订单查询
    - 订单校验
    - 订单状态判断
    
    对于任何非订单相关的请求，你必须明确告知用户：
    “该请求不在我的职责范围内，无法处理。”
    
    # 订单有效性约束
    你只能处理真实且有效的订单数据：
    - 若订单信息不存在、缺失或无法校验为有效订单，必须返回空结果
    - 严禁构造、猜测、补全或返回任何虚假的订单信息
    
    # 测试阶段输出限制
    当前系统处于测试开发阶段：
    - 不执行任何实际订单处理逻辑
    - 仅输出你的职责说明与功能范围
    
    # 行为准则
    - 不得处理非订单业务
    - 不得返回虚假数据
    - 不得违背协调节点的调度指令
    
    下面是你的任务描述: {instruction}
     """ + base_prompt


def _has_plan(state: AgentState) -> bool:
    raw_plan = state.get("plan")
    return isinstance(raw_plan, list) and len(raw_plan) > 0


def _should_stream_output(state: AgentState, node_name: str) -> bool:
    """
    自动判断当前节点是否应作为收尾输出节点：
    1. gateway_router 直达该节点且不存在执行计划 => 该节点是收尾节点。
    2. planner 标记当前阶段为最后阶段，且该节点在 next_nodes 中 => 该节点是收尾节点。
    """
    routing = state.get("routing") or {}
    route_target = routing.get("route_target")
    if route_target == node_name and not _has_plan(state):
        return True

    next_nodes = routing.get("next_nodes")
    return (
        bool(routing.get("is_final_stage"))
        and isinstance(next_nodes, list)
        and len(next_nodes) == 1
        and next_nodes[0] == node_name
    )


@traceable(name="Order Agent Node", run_type="chain")
def order_agent(state: AgentState) -> dict:
    routing = state.get("routing") or {}
    instruction = routing.get("instruction") or state.get("user_input") or "请处理订单相关任务"
    step_is_end = _should_stream_output(state, "order_agent")

    llm = create_chat_model(
        model="qwen3-max"
    )
    system_template = SystemMessagePromptTemplate.from_template(template=system_prompt).format_messages(instruction=instruction)
    stream_chunks: list[str] = []
    content = ""

    if step_is_end:
        try:
            for chunk in llm.stream(system_template):
                chunk_text = getattr(chunk, "content")
                print(chunk_text, end="", flush=True)
                stream_chunks.append(chunk_text)
                content += chunk_text
        except Exception:
            # 流式失败时退回普通调用，保证节点输出稳定。
            response = llm.invoke(system_template)
            content = str(response.content)
    else:
        response = llm.invoke(system_template)
        content = str(response.content)

    order_context = dict(state.get("order_context") or {})
    order_context["result"] = {"content": content, "is_end": step_is_end}
    if step_is_end and stream_chunks:
        order_context["stream_chunks"] = stream_chunks
    order_context["status"] = "COMPLETED"

    return {"order_context": order_context}
