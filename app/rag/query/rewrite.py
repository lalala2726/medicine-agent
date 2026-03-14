from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.rag.query.constants import RAG_REWRITE_PROMPT
from app.rag.query.utils import extract_message_content_text, normalize_question


def call_rewrite_llm(question: str) -> str:
    """使用聊天槽位模型改写检索问题。

    Args:
        question: 归一化后的原始问题。

    Returns:
        模型返回的纯文本改写结果。
    """

    llm_model = create_agent_chat_llm(
        slot=AgentChatModelSlot.CHAT,
        temperature=0.0,
        think=False,
    )
    response = llm_model.invoke(
        [
            SystemMessage(content=RAG_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
    )
    return extract_message_content_text(getattr(response, "content", ""))


def rewrite_question_for_knowledge_search(question: str) -> str:
    """将原始问题改写为更适合向量检索的查询语句。

    Args:
        question: 用户原始问题。

    Returns:
        改写后的检索语句；若改写失败则回退原始问题。
    """

    normalized_question = normalize_question(question)
    try:
        rewritten_question = call_rewrite_llm(normalized_question)
    except Exception as exc:
        logger.opt(exception=exc).warning(
            "Failed to rewrite rag question, fallback to raw question.",
        )
        return normalized_question
    rewritten_question = rewritten_question.strip()
    return rewritten_question or normalized_question
