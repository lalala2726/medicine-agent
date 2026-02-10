from typing import AsyncIterable
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, ConfigDict

from app.agent.tools.admin_tools import ADMIN_TOOLS
from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.llm import create_chat_model

ADMIN_SYSTEM_PROMPT = (
    "You are an admin assistant for the medicine mall. "
    "Help administrators manage products, orders, and operations. "
    "Use tools when needed."
)

MEMORY_STORE = InMemoryStore()
MEMORY_NAMESPACE = ("assistant", "admin")


class AdminAgentInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: list[HumanMessage]


def _store_turn(store: InMemoryStore, human_message: HumanMessage, ai_message: AIMessage) -> None:
    payload = {"human": human_message.content, "ai": ai_message.content}
    message_id = uuid4().hex

    for args in (
        (MEMORY_NAMESPACE, message_id, payload),
        (message_id, payload),
        (payload,),
    ):
        try:
            store.put(*args)
            return
        except TypeError:
            continue


async def stream_admin_assistant(question: str) -> AsyncIterable[tuple[str, bool]]:
    model = create_chat_model()
    agent = create_agent(
        model,
        tools=ADMIN_TOOLS,
        system_prompt=ADMIN_SYSTEM_PROMPT,
        store=MEMORY_STORE,
    )
    human_message = HumanMessage(content=question)

    try:
        stream = agent.astream(
            input=AdminAgentInput(messages=[human_message]),
            stream_mode="messages",
        )
        content_parts: list[str] = []

        async for chunk, metadata in stream:
            if chunk.content:
                if isinstance(chunk.content, list):
                    content = "".join(str(part) for part in chunk.content)
                else:
                    content = str(chunk.content)
                content_parts.append(content)
                yield content, False
    except ServiceException as exc:
        if exc.code == ResponseCode.UNAUTHORIZED:
            message = exc.message
        else:
            message = f"处理失败: {exc.message}"
        yield message, True
        return
    except Exception:
        yield "服务暂时不可用，请稍后重试。", True
        return

    ai_message = AIMessage(content="".join(content_parts))
    try:
        _store_turn(MEMORY_STORE, human_message, ai_message)
    except Exception:
        pass

    yield "", True
