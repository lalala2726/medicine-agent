from langchain_core.prompts import SystemMessagePromptTemplate

from app.agent.admin.agent_state import AgentState
from app.core.llm import create_chat_model

system_prompt = """
    # 系统角色定义
    你是运行在智能药品商城系统中的一个智能体节点。系统中同时存在多个智能体节点，
    你必须严格服从协调节点的统一调度与指挥，不得擅自行动或越权处理任务。
    
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
     """


def order_agent(state: AgentState) -> AgentState:
    routing = state.get("routing") or {}
    instruction = routing.get("instruction")
    if not instruction:
        # todo 未指定任务描述，这边后期将直接拒绝并将结果告诉协调器为什么未传递任务描述
        pass

    llm = create_chat_model()
    system_template = SystemMessagePromptTemplate.from_template(template=system_prompt).format_messages(instruction=instruction)
    response = llm.invoke(system_template)

    order_context = dict(state.get("order_context") or {})
    order_context["result"] = {"content": response.content}
    order_context["status"] = "COMPLETED"

    return state
