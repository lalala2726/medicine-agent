"""
MQ 基础模块包。

说明：
1. 为避免循环依赖，包级别不做聚合导入。
2. 调用方请直接按职责子包路径导入：
   - app.core.mq.config.common
   - app.core.mq.config.import_settings
   - app.core.mq.config.chunk_rebuild_settings
   - app.core.mq.config.topology
   - app.core.mq.contracts.import_models
   - app.core.mq.contracts.chunk_rebuild_models
   - app.core.mq.producers.import_command_publisher
   - app.core.mq.producers.import_result_publisher
   - app.core.mq.producers.chunk_rebuild_result_publisher
   - app.core.mq.consumers.import_consumer
   - app.core.mq.consumers.chunk_rebuild_consumer
   - app.core.mq.consumers.lifecycle
   - app.core.mq.state.import_version_store
   - app.core.mq.state.chunk_rebuild_version_store
   - app.core.mq.observability.import_logger
"""
