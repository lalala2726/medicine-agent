import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.admin.workflow import build_graph
from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.langsmith import build_langsmith_runnable_config
from app.services.assistant_stream_service import (
    AssistantStreamConfig,
    create_streaming_response,
)
from app.utils.streaming_utils import is_final_node

router = APIRouter(prefix="/admin/assistant", tags=["管理助手"])
ADMIN_WORKFLOW = build_graph()
STREAM_OUTPUT_NODES = {
    "order_agent",
    "product_agent",
    "chat_agent",
    "summary_agent",
    "chart_agent",
}


class AssistantRequest(BaseModel):
    """AI助手请求参数"""

    question: str = Field(..., description="问题")


def _invoke_admin_workflow(state: dict) -> dict:
    """
    同步执行管理助手 workflow。

    该函数用于 `astream` 不可用时的回退路径，保证接口始终可返回结果。
    """

    config = build_langsmith_runnable_config(
        run_name="admin_assistant_graph",
        tags=["admin-assistant", "langgraph"],
        metadata={"entrypoint": "api.admin_assistant.chat"},
    )
    if config:
        return ADMIN_WORKFLOW.invoke(state, config=config)
    return ADMIN_WORKFLOW.invoke(state)


def _build_stream_config() -> dict | None:
    """
    构建流式执行使用的 LangSmith 配置。

    返回 None 时表示不启用额外 tracing 配置。
    """

    return build_langsmith_runnable_config(
        run_name="admin_assistant_graph",
        tags=["admin-assistant", "langgraph"],
        metadata={"entrypoint": "api.admin_assistant.chat"},
    )


def _build_initial_state(question: str) -> dict[str, Any]:
    """
    构造管理助手的初始状态。

    所有节点共享该状态结构，避免执行过程中出现缺失键导致的分支判断复杂化。
    """

    return {
        "user_input": question,
        "user_intent": {},
        "plan": [],
        "routing": {},
        "order_context": {},
        "product_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


def _extract_content(final_state: dict[str, Any]) -> str:
    """
    从最终状态提取兜底输出文案。

    当流式过程中没有任何 token 输出时，会按业务优先级选择最终展示内容。
    """

    results = final_state.get("results") or {}
    chat_result = results.get("chat") or {}
    chat_content = chat_result.get("content")
    if isinstance(chat_content, str) and chat_content:
        return chat_content

    summary_result = results.get("summary") or {}
    summary_content = summary_result.get("content")
    if isinstance(summary_content, str) and summary_content:
        return summary_content

    chart_result = results.get("chart") or {}
    chart_content = chart_result.get("content")
    if isinstance(chart_content, str) and chart_content:
        return chart_content

    order_context = final_state.get("order_context") or {}
    order_result = order_context.get("result") or {}
    order_content = order_result.get("content")
    if isinstance(order_content, str) and order_content:
        return order_content

    product_context = final_state.get("product_context") or {}
    product_result = product_context.get("result") or {}
    product_content = product_result.get("content")
    if isinstance(product_content, str) and product_content:
        return product_content

    errors = final_state.get("errors") or []
    if errors:
        return "；".join(str(item) for item in errors)

    if results:
        return json.dumps(results, ensure_ascii=False)

    return "已完成处理。"


def _should_stream_token(stream_node: str | None, latest_state: dict[str, Any]) -> bool:
    """
    判定某个节点 token 是否应该被推送给前端。

    规则：
    1. 节点必须在可输出白名单中；
    2. chat 节点总是可输出；
    3. 其他节点只有在被判定为最终输出节点时才可输出。
    """

    if stream_node not in STREAM_OUTPUT_NODES:
        return False
    return stream_node == "chat_agent" or is_final_node(latest_state, stream_node)


def _map_exception(exc: Exception) -> str:
    """
    将内部异常映射为对用户友好的文案。

    业务异常会保留明确错误信息，未知异常统一为通用降级提示。
    """

    if isinstance(exc, ServiceException):
        return f"处理失败: {exc.message}"
    return "服务暂时不可用，请稍后重试。"


@router.post("/chat", summary="管理助手对话")
async def assistant(request: AssistantRequest) -> StreamingResponse:
    """
    管理助手聊天接口（SSE 流式返回）。

    路由层职责：
    1. 校验输入参数；
    2. 组装业务配置（AssistantStreamConfig）；
    3. 调用通用流式引擎返回 StreamingResponse。
    """

    if not request.question:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="问题不能为空")

    # 统一把“业务策略”注入流式引擎：状态构造、token 过滤、异常映射、兜底提取。
    stream_config = AssistantStreamConfig(
        workflow=ADMIN_WORKFLOW,
        build_initial_state=_build_initial_state,
        extract_final_content=_extract_content,
        should_stream_token=_should_stream_token,
        build_stream_config=_build_stream_config,
        invoke_sync=_invoke_admin_workflow,
        map_exception=_map_exception,
    )
    return create_streaming_response(request.question, stream_config)
