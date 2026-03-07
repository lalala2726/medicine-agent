"""
MQ 基础模块包。

说明：
1. 为避免循环依赖，包级别不做聚合导入。
2. 调用方请直接按职责子包路径导入：
   - app.core.mq.config.common
   - app.core.mq.config.topology
   - app.core.mq.connection
   - app.core.mq.config.document.import_settings
   - app.core.mq.config.document.chunk_rebuild_settings
   - app.core.mq.config.document.chunk_add_settings
   - app.core.mq.contracts.document.import_models
   - app.core.mq.contracts.document.chunk_rebuild_models
   - app.core.mq.contracts.document.chunk_add_models
   - app.core.mq.producers.document.import_command_publisher
   - app.core.mq.producers.document.import_result_publisher
   - app.core.mq.producers.document.chunk_rebuild_result_publisher
   - app.core.mq.producers.document.chunk_add_result_publisher
   - app.core.mq.consumers.document.import_consumer
   - app.core.mq.consumers.document.chunk_rebuild_consumer
   - app.core.mq.consumers.document.chunk_add_consumer
   - app.core.mq.consumers.lifecycle
   - app.core.mq.state.document.import_version_store
   - app.core.mq.state.document.chunk_rebuild_version_store
   - app.core.mq.observability.document.import_logger
   - app.core.mq.observability.document.chunk_rebuild_logger
   - app.core.mq.observability.document.chunk_add_logger
"""
