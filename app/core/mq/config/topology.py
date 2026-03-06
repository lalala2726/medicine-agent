"""知识库相关 MQ 拓扑常量。

当前同时覆盖两条链路：
- 知识库导入
- 单切片编辑重建
"""

# 命令与结果事件共用的交换机。
MQ_EXCHANGE_KNOWLEDGE_IMPORT = "knowledge.import"

# 智能服务消费导入命令的队列。
MQ_QUEUE_IMPORT_COMMAND = "knowledge.import.command.q"

# 业务端发布导入命令使用的路由键。
MQ_ROUTING_KEY_IMPORT_COMMAND = "knowledge.import.command"

# 智能服务发布导入结果使用的路由键。
MQ_ROUTING_KEY_IMPORT_RESULT = "knowledge.import.result"

# 切片重建命令与结果事件共用的交换机。
MQ_EXCHANGE_KNOWLEDGE_CHUNK_REBUILD = "knowledge.chunk_rebuild"

# 智能服务消费切片重建命令的队列。
MQ_QUEUE_CHUNK_REBUILD_COMMAND = "knowledge.chunk_rebuild.command.q"

# 业务端发布切片重建命令使用的路由键。
MQ_ROUTING_KEY_CHUNK_REBUILD_COMMAND = "knowledge.chunk_rebuild.command"

# 智能服务发布切片重建结果使用的路由键。
MQ_ROUTING_KEY_CHUNK_REBUILD_RESULT = "knowledge.chunk_rebuild.result"
