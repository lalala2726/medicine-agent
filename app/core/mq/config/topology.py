"""知识库导入流程的 RabbitMQ 拓扑常量。"""

# 命令与结果事件共用的交换机。
MQ_EXCHANGE_KNOWLEDGE_IMPORT = "knowledge.import"

# 智能服务消费导入命令的队列。
MQ_QUEUE_IMPORT_COMMAND = "knowledge.import.command.q"

# 业务端发布导入命令使用的路由键。
MQ_ROUTING_KEY_IMPORT_COMMAND = "knowledge.import.command"

# 智能服务发布导入结果使用的路由键。
MQ_ROUTING_KEY_IMPORT_RESULT = "knowledge.import.result"
