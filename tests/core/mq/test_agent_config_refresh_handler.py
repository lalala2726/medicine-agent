"""agent_config_refresh_handler 消费者单元测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.core.mq.models.agent_config_refresh import AgentConfigRefreshMessage

_HANDLER_MODULE = "app.core.mq.handlers.agent_config_refresh_handler"

_VALID_MESSAGE = AgentConfigRefreshMessage(
    redis_key="agent:config:all",
    updated_at="2026-03-11T14:30:00+08:00",
    updated_by="admin",
    created_at="2026-03-11T14:30:01+08:00",
)


def test_handle_agent_config_refresh_passes_redis_key() -> None:
    """测试目的：刷新消费者应把 Redis key 透传给快照刷新逻辑；预期结果：调用参数匹配消息体。"""

    with patch(f"{_HANDLER_MODULE}.refresh_agent_config_snapshot") as mock_refresh:
        from app.core.mq.handlers.agent_config_refresh_handler import handle_agent_config_refresh

        asyncio.run(handle_agent_config_refresh(_VALID_MESSAGE))

    mock_refresh.assert_called_once_with(
        redis_key="agent:config:all",
    )
