from typing import Any, Literal

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


class HistoryMessageRequest(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")


class AssistantRequest(BaseModel):
    """AI助手请求参数"""

    question: str = Field(..., description="问题")
    history_messages: list[HistoryMessageRequest] = Field(
        default_factory=list,
        description="可选历史消息，元素为 {role, content}",
    )


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


def _build_initial_state(
        question: str,
        history_messages: list[HistoryMessageRequest] | None = None,
) -> dict[str, Any]:
    """
    构造管理助手的初始状态。

    所有节点共享该状态结构，避免执行过程中出现缺失键导致的分支判断复杂化。
    """

    # 初始状态里预置 step_outputs/history_messages，
    # 让 DAG 节点与 planner 不需要做大量判空分支。
    return {
        "user_input": question,
        "user_intent": {},
        "plan": [],
        "routing": {},
        "order_context": {},
        "product_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "history_messages": [
            {"role": item.role, "content": item.content}
            for item in (history_messages or [])
        ],
        "step_outputs": {},
        "shared_memory": {},
        "results": {},
        "errors": [],
    }


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

    # 统一把“业务策略”注入流式引擎：状态构造、token 过滤、异常映射。
    # 管理助手不做 extract_content 兜底：当无最终 token 输出时，不补发 answer 文本。
    stream_config = AssistantStreamConfig(
        workflow=ADMIN_WORKFLOW,
        build_initial_state=lambda question: _build_initial_state(
            question,
            request.history_messages,
        ),
        extract_final_content=lambda _state: "",
        should_stream_token=_should_stream_token,
        build_stream_config=_build_stream_config,
        invoke_sync=_invoke_admin_workflow,
        map_exception=_map_exception,
    )
    return create_streaming_response(request.question, stream_config)
