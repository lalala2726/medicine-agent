"""
管理端单 Agent 工具包。

说明：
1. 该包集中管理 admin 单 Agent 的基础工具、业务工具、动态注入中间件与注册中心；
2. 业务工具按领域拆分到独立模块，避免再次回到多 domain 节点结构；
3. 会话级工具缓存也由本包统一提供；
4. 对外仅暴露注册中心、动态中间件与缓存入口，节点层不直接依赖旧 domain 目录。
"""

from app.agent.admin.tools.cache import (
    ADMIN_TOOL_CACHE_KEY_PREFIX,
    ADMIN_TOOL_CACHE_MAX_PROMPT_RECORDS,
    ADMIN_TOOL_CACHE_PROMPT_TITLE,
    ADMIN_TOOL_CACHE_TTL_SECONDS,
    bind_current_admin_tool_cache_conversation,
    build_admin_tool_cache_key,
    load_admin_tool_cache,
    render_admin_tool_cache_prompt,
    reset_current_admin_tool_cache_conversation,
    save_admin_tool_cache_entry,
    save_current_admin_tool_cache_entry,
)
from app.agent.admin.tools.middleware import AdminDynamicToolMiddleware
from app.agent.admin.tools.registry import ADMIN_TOOL_REGISTRY, AdminToolRegistry

__all__ = [
    "ADMIN_TOOL_CACHE_KEY_PREFIX",
    "ADMIN_TOOL_CACHE_MAX_PROMPT_RECORDS",
    "ADMIN_TOOL_CACHE_PROMPT_TITLE",
    "ADMIN_TOOL_CACHE_TTL_SECONDS",
    "ADMIN_TOOL_REGISTRY",
    "AdminDynamicToolMiddleware",
    "AdminToolRegistry",
    "bind_current_admin_tool_cache_conversation",
    "build_admin_tool_cache_key",
    "load_admin_tool_cache",
    "render_admin_tool_cache_prompt",
    "reset_current_admin_tool_cache_conversation",
    "save_admin_tool_cache_entry",
    "save_current_admin_tool_cache_entry",
]
