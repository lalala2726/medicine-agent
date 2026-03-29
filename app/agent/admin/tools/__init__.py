"""
管理端单 Agent 工具包。

说明：
1. 该包集中管理 admin 单 Agent 的基础工具、业务工具、动态注入中间件与注册中心；
2. 业务工具按领域拆分到独立模块，避免再次回到多 domain 节点结构；
3. 对外仅暴露注册中心与动态注入中间件，统一缓存走 `app.core.agent.tool_cache`。
"""

from app.agent.admin.tools.middleware import AdminDynamicToolMiddleware
from app.agent.admin.tools.registry import ADMIN_TOOL_REGISTRY, AdminToolRegistry

__all__ = [
    "ADMIN_TOOL_REGISTRY",
    "AdminDynamicToolMiddleware",
    "AdminToolRegistry",
]
