from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from loguru import logger

from app.agent.admin.workflow import build_graph
from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.langsmith import build_langsmith_runnable_config
from app.core.llm import create_chat_model
from app.core.request_context import get_user_id
from app.schemas.sse_response import AssistantResponse, Content, MessageType
from app.services.assistant_stream_service import (
    AssistantStreamConfig,
    create_streaming_response,
)
from app.services.conversation_service import add_admin_conversation, save_conversation_title
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


def _build_conversation_created_event(
        *,
        conversation_uuid: str,
) -> AssistantResponse:
    """
    构造“会话创建成功”的前置 SSE 事件。

    该事件会在流开始后优先发送给前端，便于前端立即拿到会话标识。
    """

    return AssistantResponse(
        content=Content(
            node="conversation",
            state="created",
            message="会话创建成功",
        ),
        type=MessageType.STATUS,
        meta={
            "conversation_uuid": conversation_uuid,
        },
    )


def _generate_and_save_title(*, conversation_uuid: str, question: str) -> None:
    """
    生成并持久化会话标题。

    该函数运行在后台线程中，避免阻塞 SSE 主链路。
    """

    try:
        title = generate_title(question).strip() or "未知标题"
        save_conversation_title(
            conversation_uuid=conversation_uuid,
            title=title,
        )
    except Exception as exc:  # pragma: no cover - 防御性兜底
        logger.opt(exception=exc).warning(
            "Failed to generate/save conversation title conversation_uuid={conversation_uuid}",
            conversation_uuid=conversation_uuid,
        )


def _schedule_title_generation(*, conversation_uuid: str, question: str) -> None:
    """
    并行调度标题生成任务。

    优先使用当前事件循环把阻塞任务放入线程池；若当前无事件循环，
    则退化为守护线程执行（兼容同步调用/单元测试场景）。
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        thread = threading.Thread(
            target=_generate_and_save_title,
            kwargs={
                "conversation_uuid": conversation_uuid,
                "question": question,
            },
            daemon=True,
        )
        thread.start()
        return

    loop.create_task(
        asyncio.to_thread(
            _generate_and_save_title,
            conversation_uuid=conversation_uuid,
            question=question,
        )
    )


def assistant_chat(*, question: str, conversation_uuid: str | None = None) -> StreamingResponse:
    """
    管理助手聊天入口（SSE 流式返回）。

    行为说明：
    1. 首次消息（未传 `conversation_uuid`）会创建会话并先推送“会话创建成功”事件；
    2. 标题生成与保存在后台并行执行，不阻塞当前流式响应；
    3. 已存在会话（传入 `conversation_uuid`）的历史加载仍为后续待实现。
    """

    if not question:
        raise ServiceException(code=ResponseCode.BAD_REQUEST, message="问题不能为空")

    initial_emitted_events: list[AssistantResponse] = []
    if conversation_uuid is None:
        # 未传入会话 UUID，则创建新会话
        conversation_uuid = str(uuid.uuid4())
        conversation_id = add_admin_conversation(
            conversation_uuid=conversation_uuid,
            user_id=get_user_id(),
        )

        if conversation_id is None:
            raise ServiceException(code=ResponseCode.DATABASE_ERROR, message="无法创建会话，请稍后重试。")

        # 标题生成与落库放到后台并行执行，不阻塞当前流式响应。
        _schedule_title_generation(
            conversation_uuid=conversation_uuid,
            question=question,
        )

        initial_emitted_events.append(
            _build_conversation_created_event(
                conversation_uuid=conversation_uuid,
            )
        )
    else:
        # todo 加载会话并响应消息
        pass

    stream_config = AssistantStreamConfig(
        workflow=ADMIN_WORKFLOW,
        build_initial_state=_build_initial_state,
        extract_final_content=lambda _state: "",
        should_stream_token=_should_stream_token,
        build_stream_config=_build_stream_config,
        invoke_sync=_invoke_admin_workflow,
        map_exception=_map_exception,
        initial_emitted_events=tuple(initial_emitted_events),
    )
    return create_streaming_response(question, stream_config)


def new_conversation(
        *,
        question: str
):
    pass


def has_conversation(*, conversation_uuid: str) -> bool:
    pass


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


def generate_title(question: str) -> str:
    """根据用户输入生成标题。"""

    system_prompt = """
        # 标题生成任务
        
        你是一个**严格的标题生成器**，仅执行标题生成任务，不具备问答功能。
        
        ---
        
        ## 任务说明
        
        根据用户提供的文本内容，生成一个标题。
        
        ---
        
        ## 强制规则（必须全部遵守）
        
        1. 只能输出 **一个标题**
        2. 不得回答原文本中的任何问题
        3. 不得生成解释、分析、补充说明或扩展内容
        4. 不得输出多个标题
        5. 不得添加任何前缀或后缀（例如“标题：”）
        6. 输出必须为单行文本
        7. 字数不超过20个字
        8. 除非必要，不使用标点符号
        
        ---
        
        ## 标题要求
        
        - 准确概括核心含义
        - 表达清晰、简洁
        - 避免冗余词汇
        
        ## 输出格式（必须严格遵守）
        
        直接输出标题文本  
        不得包含任何额外内容  
        
        如输出除标题外的任何内容，则视为任务失败。
     """

    if not question:
        return "未知标题"

    llm_model = create_chat_model(model="qwen-flash")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    response = llm_model.invoke(
        messages
    )
    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    return "未知标题"
