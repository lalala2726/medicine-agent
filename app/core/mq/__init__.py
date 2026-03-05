"""
MQ 基础模块包。

说明：
1. 为避免循环依赖，包级别不做聚合导入。
2. 调用方请直接按职责子包路径导入：
   - app.core.mq.config.settings
   - app.core.mq.config.topology
   - app.core.mq.contracts.models
   - app.core.mq.producers.command_publisher
   - app.core.mq.producers.result_publisher
   - app.core.mq.consumers.import_consumer
   - app.core.mq.consumers.lifecycle
   - app.core.mq.state.latest_version_store
   - app.core.mq.observability.import_logger
"""
