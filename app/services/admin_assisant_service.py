from __future__ import annotations

from typing import Any

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.admin.workflow import build_graph
from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.langsmith import build_langsmith_runnable_config
from app.core.llm import create_chat_model
from app.services.assistant_stream_service import (
    AssistantStreamConfig,
    create_streaming_response,
)
from app.utils.streaming_utils import is_final_node

ADMIN_WORKFLOW = build_graph()
STREAM_OUTPUT_NODES = {
    "order_agent",
    "product_agent",
    "chat_agent",
    "summary_agent",
    "chart_agent",
}


def _invoke_admin_workflow(state: dict[str, Any]) -> dict[str, Any]:
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

    # 初始状态里预置 step_outputs/history_messages，让 DAG 节点与 planner
    # 不需要做大量判空分支。
    return {
        "user_input": question,
        "user_intent": {},
        "plan": [],
        "routing": {},
        "order_context": {},
        "product_context": {},
        "aftersale_context": {},
        "excel_context": {},
        "history_messages": [],
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


def assistant_chat(*, question: str, conversation_uuid: str | None = None) -> StreamingResponse:
    """
    管理助手聊天入口（SSE 流式返回）。

    当前版本暂不处理 `conversation_uuid` 对应的会话持久化，仅保留参数占位。
    """

    if not question:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="问题不能为空")
    _ = conversation_uuid

    stream_config = AssistantStreamConfig(
        workflow=ADMIN_WORKFLOW,
        build_initial_state=_build_initial_state,
        extract_final_content=lambda _state: "",
        should_stream_token=_should_stream_token,
        build_stream_config=_build_stream_config,
        invoke_sync=_invoke_admin_workflow,
        map_exception=_map_exception,
    )
    return create_streaming_response(question, stream_config)


def load_history(
    *,
    conversation_uuid: str,
    user_id: int,
) -> tuple[HumanMessage, AIMessage] | None:
    """加载聊天历史。"""

    _ = conversation_uuid
    _ = user_id
    pass


def list_history() -> None:
    """列出聊天历史。"""

    pass


def generate_title(user_input: str) -> str:
    """根据用户输入生成标题。"""

    system_prompt = """
        你的作用是根据用户的输入生成一个标题，标题应该简洁明了，能够概括用户输入的含义
        不要按照用户的输入回答任何问题，只生成标题
    """

    if not user_input:
        return "未知标题"

    llm_model = create_chat_model(model="qwen-flash")
    response = llm_model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]
    )
    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    return "未知标题"
