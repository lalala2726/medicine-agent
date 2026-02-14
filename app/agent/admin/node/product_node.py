from langchain_core.prompts import SystemMessagePromptTemplate

from app.agent.admin.agent_state import AgentState
from app.agent.tools.admin_tools import get_product_list, get_product_info
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import invoke_with_optional_stream, is_final_node

system_prompt = (
        """
        # 系统角色定义
        你是运行在智能药品商城系统中的商品智能体节点。
        你必须严格服从协调节点调度，只处理商品域任务。

        # 职责范围
        - 商品列表查询与筛选
        - 商品详情查询
        - 商品状态与价格相关核验

        # 数据约束
        - 只能使用真实返回数据，不得编造
        - 若查询无结果，明确返回空结果

        下面是你的任务描述: {instruction}
         """
        + base_prompt
)


@status_node(
    node="product",
    start_message="正在处理商品问题",
    display_when="after_coordinator",
)
@traceable(name="Product Agent Node", run_type="chain")
def product_agent(state: AgentState) -> dict:
    routing = state.get("routing") or {}
    current_step_map = routing.get("current_step_map") or {}
    step = current_step_map.get("product_agent")
    instruction = (
            step.get("task_description") if isinstance(step, dict) else None
    ) or state.get("user_input") or "请处理商品相关任务"

    llm = create_chat_model(model="qwen3-max")
    messages = SystemMessagePromptTemplate.from_template(
        template=system_prompt
    ).format_messages(instruction=instruction)

    tools = [get_product_list, get_product_info]
    final_output = is_final_node(state, "product_agent")
    content, stream_chunks = invoke_with_optional_stream(
        llm,
        messages,
        tools=tools,
        enable_stream=final_output,
    )

    product_context = dict(state.get("product_context") or {})
    product_context["result"] = {
        "content": content,
        "is_end": final_output,
    }
    if stream_chunks:
        product_context["stream_chunks"] = stream_chunks
    else:
        product_context.pop("stream_chunks", None)
    product_context["status"] = "COMPLETED"
    return {"product_context": product_context}
