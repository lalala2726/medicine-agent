# `app/core/mq`

RabbitMQ 基础设施层，负责 MQ 通讯公共能力：配置、协议、收发、状态判定、可观测性。

业务处理逻辑在 `app/services/`（`knowledge_base_service.py`、`chunk_rebuild_service.py`）。

## 目录结构

```
app/core/mq/
├── _aio_pika_loader.py                         # aio-pika 延迟加载（全局共享）
├── config/
│   ├── common.py                               # 公共工具：开关判断、参数解析
│   ├── topology.py                             # exchange / queue / routing key 常量
│   ├── import_settings.py                      # 知识库导入 MQ 配置
│   └── chunk_rebuild_settings.py               # 切片重建 MQ 配置
├── contracts/
│   ├── import_models.py                        # 导入链路 command / result 消息模型
│   └── chunk_rebuild_models.py                 # 切片重建 command / result 消息模型
├── producers/
│   ├── import_command_publisher.py             # 发送导入命令（AI → 业务）
│   ├── import_result_publisher.py              # 发送导入结果（AI → 业务）
│   └── chunk_rebuild_result_publisher.py       # 发送切片重建结果（AI → 业务）
├── consumers/
│   ├── import_consumer.py                      # 导入命令消费与流程编排
│   ├── chunk_rebuild_consumer.py               # 切片重建命令消费与流程编排
│   └── lifecycle.py                            # FastAPI lifespan 消费者启停管理
├── state/
│   ├── _version_store_support.py               # Redis 版本读取公共逻辑
│   ├── import_version_store.py                 # 导入链路 latest-version 判定
│   └── chunk_rebuild_version_store.py          # 切片重建 latest-version 判定
└── observability/
    ├── import_logger.py                        # 导入链路结构化日志
    └── chunk_rebuild_logger.py                 # 切片重建链路结构化日志
```

## 两条 MQ 链路

### 知识库导入

业务侧发送导入命令 → AI 侧下载、解析、切片、向量化 → 写入 Milvus → 返回结果。

- 协议：`contracts/import_models.py`
- 消费：`consumers/import_consumer.py`
- 状态：`state/import_version_store.py`（按 `biz_key` 粒度丢弃旧版本）
- ACK：处理结束后统一 ACK，结果投递失败不阻塞 ACK

### 切片重建

业务侧编辑切片内容 → AI 侧重新向量化 → 更新 Milvus 单行 → 返回结果。

- 协议：`contracts/chunk_rebuild_models.py`
- 消费：`consumers/chunk_rebuild_consumer.py`
- 状态：`state/chunk_rebuild_version_store.py`（按 `vector_id` 粒度，消费前 + 写入前各判定一次）
- ACK：结果消息成功投递后才 ACK，投递失败则保留给 MQ 重投

## 新增链路参考

1. `config/` — 新增 exchange / queue / routing key 配置
2. `contracts/` — 定义 command / result 消息模型
3. `producers/` — 实现结果消息发布
4. `app/services/` — 实现业务处理逻辑
5. `consumers/` — 实现消费编排（解析 → 调用 service → 发结果 → 控制 ACK）
6. `state/` — 按需增加 Redis latest-version 判定
7. `observability/` — 新增结构化日志（参考 `import_logger.py`）
8. `consumers/lifecycle.py` — 接入应用启停

## 相关文档

- 切片重建 MQ 对接文档：`docs/knowledge_chunk_rebuild_mq_api.md`
