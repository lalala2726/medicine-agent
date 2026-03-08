"""MQ 拓扑定义：Exchange、Queue、Routing Key。

所有交换机和队列均为 **durable**，使用 **DIRECT** 类型。
拓扑名称与业务服务约定，修改时需同步两端。
"""

from __future__ import annotations

from faststream.rabbit import ExchangeType, RabbitExchange, RabbitQueue

# ---- 知识库导入 ----------------------------------------------------------------

# 导入链路交换机（DIRECT，durable）
import_exchange = RabbitExchange(
    "knowledge.import", type=ExchangeType.DIRECT, durable=True,
)
# 导入命令队列，绑定 routing_key = knowledge.import.command
import_command_queue = RabbitQueue(
    "knowledge.import.command.q",
    routing_key="knowledge.import.command",
    durable=True,
)
# 导入结果路由键（AI → 业务）
IMPORT_RESULT_ROUTING_KEY = "knowledge.import.result"

# ---- 切片重建 --------------------------------------------------------------------

# 切片重建链路交换机
chunk_rebuild_exchange = RabbitExchange(
    "knowledge.chunk_rebuild", type=ExchangeType.DIRECT, durable=True,
)
# 切片重建命令队列
chunk_rebuild_command_queue = RabbitQueue(
    "knowledge.chunk_rebuild.command.q",
    routing_key="knowledge.chunk_rebuild.command",
    durable=True,
)
# 切片重建结果路由键（AI → 业务）
CHUNK_REBUILD_RESULT_ROUTING_KEY = "knowledge.chunk_rebuild.result"

# ---- 手工新增切片 ----------------------------------------------------------------

# 手工新增切片链路交换机
chunk_add_exchange = RabbitExchange(
    "knowledge.chunk_add", type=ExchangeType.DIRECT, durable=True,
)
# 手工新增切片命令队列
chunk_add_command_queue = RabbitQueue(
    "knowledge.chunk_add.command.q",
    routing_key="knowledge.chunk_add.command",
    durable=True,
)
# 手工新增切片结果路由键（AI → 业务）
CHUNK_ADD_RESULT_ROUTING_KEY = "knowledge.chunk_add.result"
