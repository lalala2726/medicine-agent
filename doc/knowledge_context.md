# 知识库功能上下文记录（对话沉淀）

> 更新时间：2026-03-03  
> 目的：避免后续准备功能开发时丢失上下文，快速恢复需求背景与关键决策。

## 1. 你当前想做的事情（核心目标）

你要做的是一套“知识库管理 + 文档导入 + 切片检索”的完整能力，前端需要支持以下流程：

1. 查看已创建的知识库列表。
2. 进入某个知识库后查看上传文件列表。
3. 点击文件查看该文件的切片列表。
4. 点击切片查看切片详情。

你现在处于“先做基础准备”的阶段，先把模型、边界、数据库结构定稳，再逐步实现功能。

## 2. 系统架构边界（已确认）

### 2.1 职责拆分

1. SpringBoot 负责业务主流程与文件上传入口（并维护 MySQL 业务主数据）。
2. Python（本项目）负责文档解析、切片、向量化、检索相关能力，通过 HTTP 被 SpringBoot 调用。
3. 向量/切片正文数据放 Milvus；MySQL 不重复存完整切片正文与向量。

### 2.2 数据真源

1. 业务主数据真源：MySQL（由 SpringBoot 维护）。
2. 检索数据真源：Milvus（chunk + vector）。
3. Python 不做业务主数据真源，只做处理引擎。

## 3. 切片模块相关结论（已落地）

你确认了 `chunking` 应该聚焦“文本切片”单一职责，文件路径解析不属于 chunking。  
已实现方向：

1. 输入：`text + strategy + config`
2. 输出：`List[SplitChunk]`
3. 策略类型使用枚举，不用裸常量字符串。
4. 枚举值改为：`character`、`recursive`、`token`、`markdown_header`。

## 4. MySQL 元数据建模决策（已确认）

### 4.1 本次建表范围

一次创建 4 张核心表：

1. `knowledge_base`
2. `knowledge_document`
3. `knowledge_import_task`
4. `knowledge_document_index`

### 4.2 统一规范

1. MySQL 8.0。
2. 主键：`BIGINT UNSIGNED AUTO_INCREMENT`。
3. 所有表/字段必须有注释。
4. 软删除：`is_deleted` + `deleted_at`。
5. 审计字段：`create_by`、`update_by`、`created_at`、`updated_at`。
6. 启用外键约束（`RESTRICT`）。
7. 文档去重走数据库唯一约束。

### 4.3 去重与状态

1. 文档去重规则：`(knowledge_base_id, file_sha256, is_deleted)` 唯一。
2. 导入任务状态：`PENDING / RUNNING / SUCCESS / FAILED`。
3. 文档版本策略：暂不做完整版本化，但预留版本字段。

## 5. 已产出文件

1. DDL 文件：`doc/knowledge_schema.sql`
2. 内容包括：
    1. 4 张表完整建表语句；
    2. 表注释和字段注释；
    3. 索引、唯一键、外键；
    4. 联调示例查询（知识库列表、文件列表、任务列表、索引快照）。

## 6. 后续实现建议顺序（下一步参考）

1. SpringBoot 先落库这 4 张表，并实现知识库/文件/任务的基础 CRUD。
2. 先打通“上传文件 -> 调 Python 导入 -> 回写任务状态”的最小闭环。
3. 再补“文件列表/切片列表/切片详情”页面接口。
4. 最后完善重试、幂等、失败补偿、权限和审计。

## 7. 关键约定（避免后续跑偏）

1. `document_id` 对齐 `knowledge_document.id`，并透传给 Python。
2. `knowledge_name` 由 `knowledge_base.knowledge_name` 统一管理。
3. MySQL 不建 `knowledge_chunk`（当前阶段），切片详情从 Milvus 查询。
4. 若后续需要切片审计或人工修订，再评估加轻量 chunk 映射表。

## 8. URL 下载落盘新约定（2026-03-03 新增）

1. URL 下载不再使用系统临时目录，改为固定目录落盘。
2. 固定目录通过 `.env` 的 `FILE_DOWNLOAD_ROOT_DIR` 提供，且为必填项；未配置直接报错。
3. 落盘目录规则：`yyyy/mm/dd`。
4. 落盘文件名规则：`uuid_原文件名`（文件名会做安全清洗）。
5. 当前阶段不实现自动清理任务。
6. 下载文件成功与失败都保留，便于后续排障与审计。

## 9. 文件解析与切片新约定（去分页重写，2026-03-03 新增）

1. 文件解析结果不再返回 `pages`，统一只返回单一 `text` 字段。
2. 旧分页字段 `page_number/page_label` 全部移除，不做兼容。
3. 切片输出统一为对象列表：`[{text, stats:{char_count}}]`。
4. 旧切片字段 `chunk_index/metadata` 全部移除，不做兼容。
5. 导入链路固定为：URL 校验 -> 下载 -> 类型识别 -> 解析文本 -> 文本清洗 -> 按策略切片。
6. 当前调试阶段切片结果打印到控制台，不写入向量库。

## 10. HTTP 建库与向量字段定稿（2026-03-03 新增）

1. 建库接口保持 `POST /knowledge_base`，请求体仅包含 `knowledge_name`、`embedding_dim`、`description`。
2. `embedding_dim` 继续由 HTTP 传入，不在 Python 内做固定维度硬编码。
3. Milvus 使用“每知识库一集合”模型，集合名使用 `knowledge_name`。
4. Milvus 字段定稿为 11 个：`id`、`document_id`、`chunk_no`、`content`、`char_count`、`embedding`、`chunk_strategy`、
   `chunk_size`、`token_size`、`source_hash`、`created_at_ts`。
5. 向量索引保持 `embedding -> AUTOINDEX + COSINE`，标量索引保持 `document_id -> STL_SORT`。
6. `embedding_model` 由 SpringBoot/MySQL 元数据层管理（`knowledge_base`、`knowledge_document_index`），Python 建库接口不接收该字段。
