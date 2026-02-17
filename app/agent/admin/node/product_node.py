from langchain_core.prompts import SystemMessagePromptTemplate

from app.agent.admin.agent_utils import invoke_with_failure_policy
from app.agent.admin.agent_state import AgentState
from app.agent.admin.node.runtime_context import (
    build_instruction_with_failure_policy,
    build_step_output_update,
    build_step_runtime,
    evaluate_failure_by_policy,
)
from app.agent.tools.admin_tools import get_product_list, get_product_detail
from app.core.assistant_status import status_node
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.utils.streaming_utils import is_final_node

system_prompt = (
        """
        # 系统角色定义
        你是运行在智能药品商城系统中的商品智能体节点。
        你必须严格服从协调节点调度，只处理商品域任务。

        # 职责范围
        - 商品列表查询与筛选
        - 商品详情查询
        - 商品状态与价格相关核验
        大部分情况下你在调用商品列表商品列表给你返回的信息就可以获取大部分信息，除非用户的需求商品列表无法满足你可以调用商品详情获取更加详细的信息
        
        # 工具使用
        - 商品列表查询：调用 get_product_list 工具，根据商品名称、分类、价格区间、品牌等条件进行商品列表查询。
        - 商品详情查询：调用 get_product_detail 工具，根据商品 ID 获取商品详情信息。当你需要查询多个商品详情时，这边优先一次性传递多个商品 ID 进行批量查询。
            应该尽量避免一次传递1个商品 ID，优先一次性传递多个商品 ID 进行批量查询。

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
    # 从 planner 当前步骤中读取运行时配置与上游可读输出。
    runtime = build_step_runtime(
        state,
        "product_agent",
        default_task_description="请处理商品相关任务",
    )
    # 将任务描述 + 上游输出 + 可选上下文整合，并统一附加失败策略提示。
    instruction = build_instruction_with_failure_policy(runtime)
    failure_policy = runtime.get("failure_policy") or {}

    llm = create_chat_model(model="qwen3-max")
    messages = SystemMessagePromptTemplate.from_template(
        template=system_prompt
    ).format_messages(instruction=instruction)

    tools = [get_product_list, get_product_detail]
    # 只有最终输出步骤才开启 stream 分支。
    final_output = is_final_node(state, "product_agent")
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
        content = "商品服务暂时不可用，请稍后重试。"
        stream_chunks = []
        step_status = "failed"
        failed_error = f"product_agent 执行失败: {exc}"

    # 兼容保留原 product_context 输出结构。
    product_context = dict(state.get("product_context") or {})
    product_context["result"] = {
        "content": content,
        "is_end": final_output,
    }
    if stream_chunks:
        product_context["stream_chunks"] = stream_chunks
    else:
        product_context.pop("stream_chunks", None)
    product_context["status"] = "COMPLETED" if step_status == "completed" else "FAILED"

    # 写入 step_outputs，驱动 DAG 调度与故障传播。
    result = {"product_context": product_context}
    result.update(
        build_step_output_update(
            runtime,
            node_name="product_agent",
            status=step_status,
            text=content,
            output={"result": product_context["result"]},
            error=failed_error,
        )
    )
    return result
