-- ============================================================
-- Knowledge Metadata Schema (MySQL 8.0)
-- Path: doc/knowledge_schema.sql
-- Notes:
-- 1) This schema stores business metadata only.
-- 2) Chunk detail and vectors stay in Milvus.
-- 3) Soft delete is enabled for all tables.
-- ============================================================

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ------------------------------------------------------------
-- Table: knowledge_base
-- Purpose: knowledge base master metadata for list pages
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `knowledge_base` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  `knowledge_name` VARCHAR(128) NOT NULL COMMENT '知识库唯一名称（业务键）',
  `display_name` VARCHAR(255) NOT NULL COMMENT '知识库展示名称',
  `description` VARCHAR(1024) NOT NULL DEFAULT '' COMMENT '知识库描述',
  `milvus_collection_name` VARCHAR(128) NOT NULL COMMENT '绑定的 Milvus 集合名称',
  `embedding_model` VARCHAR(128) NOT NULL DEFAULT '' COMMENT '向量模型标识',
  `embedding_dim` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '向量维度',
  `status` VARCHAR(32) NOT NULL DEFAULT 'ACTIVE' COMMENT '记录状态，例如 ACTIVE 或 DISABLED',
  `create_by` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '创建人账号',
  `update_by` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '最后更新人账号',
  `created_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `updated_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  `is_deleted` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '逻辑删除标记：0 未删除，1 已删除',
  `deleted_at` DATETIME(3) NULL DEFAULT NULL COMMENT '逻辑删除时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_kb_name_deleted` (`knowledge_name`, `is_deleted`),
  KEY `idx_kb_status_deleted` (`status`, `is_deleted`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='知识库主数据表';

-- ------------------------------------------------------------
-- Table: knowledge_document
-- Purpose: uploaded file metadata under a knowledge base
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `knowledge_document` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  `knowledge_base_id` BIGINT UNSIGNED NOT NULL COMMENT '知识库ID（关联 knowledge_base.id）',
  `file_name` VARCHAR(255) NOT NULL COMMENT '原始文件名',
  `file_url` VARCHAR(2048) NOT NULL COMMENT '文件 URL',
  `file_sha256` CHAR(64) NOT NULL COMMENT '文件内容 SHA-256 哈希',
  `file_size` BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '文件大小（字节）',
  `mime_type` VARCHAR(255) NOT NULL DEFAULT '' COMMENT '文件 MIME 类型',
  `version` INT UNSIGNED NOT NULL DEFAULT 1 COMMENT '预留文档版本字段',
  `status` VARCHAR(16) NOT NULL DEFAULT 'PENDING' COMMENT '索引状态：PENDING、RUNNING、SUCCESS、FAILED',
  `last_error` VARCHAR(2000) NOT NULL DEFAULT '' COMMENT '最近一次处理失败错误信息',
  `create_by` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '创建人账号',
  `update_by` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '最后更新人账号',
  `created_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `updated_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  `is_deleted` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '逻辑删除标记：0 未删除，1 已删除',
  `deleted_at` DATETIME(3) NULL DEFAULT NULL COMMENT '逻辑删除时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_doc_kb_sha_deleted` (`knowledge_base_id`, `file_sha256`, `is_deleted`),
  KEY `idx_doc_kb_status` (`knowledge_base_id`, `status`, `is_deleted`),
  CONSTRAINT `fk_doc_kb` FOREIGN KEY (`knowledge_base_id`) REFERENCES `knowledge_base` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='知识库文档元数据表';

-- ------------------------------------------------------------
-- Table: knowledge_import_task
-- Purpose: import execution tracking and status history
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `knowledge_import_task` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  `knowledge_base_id` BIGINT UNSIGNED NOT NULL COMMENT '知识库ID（关联 knowledge_base.id）',
  `document_id` BIGINT UNSIGNED NOT NULL COMMENT '文档ID（关联 knowledge_document.id）',
  `task_uuid` CHAR(36) NOT NULL COMMENT '调用方生成的任务 UUID',
  `trigger_source` VARCHAR(32) NOT NULL DEFAULT 'SPRINGBOOT_HTTP' COMMENT '任务触发来源',
  `status` VARCHAR(16) NOT NULL DEFAULT 'PENDING' COMMENT '任务状态：PENDING、RUNNING、SUCCESS、FAILED',
  `progress` DECIMAL(5,2) NOT NULL DEFAULT 0.00 COMMENT '任务进度百分比 [0,100]',
  `chunk_strategy` VARCHAR(32) NOT NULL DEFAULT 'character' COMMENT '切片策略快照',
  `chunk_size` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '字符切片大小快照',
  `token_size` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Token 切片大小快照',
  `chunk_overlap` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '切片重叠大小快照',
  `started_at` DATETIME(3) NULL DEFAULT NULL COMMENT '任务开始时间',
  `finished_at` DATETIME(3) NULL DEFAULT NULL COMMENT '任务结束时间',
  `error_message` VARCHAR(2000) NOT NULL DEFAULT '' COMMENT '任务失败错误详情',
  `create_by` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '创建人账号',
  `update_by` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '最后更新人账号',
  `created_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `updated_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  `is_deleted` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '逻辑删除标记：0 未删除，1 已删除',
  `deleted_at` DATETIME(3) NULL DEFAULT NULL COMMENT '逻辑删除时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_task_uuid_deleted` (`task_uuid`, `is_deleted`),
  KEY `idx_task_doc_status` (`document_id`, `status`, `is_deleted`),
  KEY `idx_task_kb_created` (`knowledge_base_id`, `created_at`),
  CONSTRAINT `fk_task_kb` FOREIGN KEY (`knowledge_base_id`) REFERENCES `knowledge_base` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `fk_task_doc` FOREIGN KEY (`document_id`) REFERENCES `knowledge_document` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='知识库导入任务表';

-- ------------------------------------------------------------
-- Table: knowledge_document_index
-- Purpose: index snapshot metadata to align document and vector index
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `knowledge_document_index` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  `knowledge_base_id` BIGINT UNSIGNED NOT NULL COMMENT '知识库ID（关联 knowledge_base.id）',
  `document_id` BIGINT UNSIGNED NOT NULL COMMENT '文档ID（关联 knowledge_document.id）',
  `index_version` INT UNSIGNED NOT NULL DEFAULT 1 COMMENT '预留索引版本字段',
  `index_status` VARCHAR(16) NOT NULL DEFAULT 'PENDING' COMMENT '索引状态：PENDING、RUNNING、SUCCESS、FAILED',
  `chunk_count` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '本次索引生成的切片数量',
  `token_count` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '本次索引生成的 Token 数量',
  `chunk_strategy` VARCHAR(32) NOT NULL DEFAULT 'character' COMMENT '切片策略快照',
  `chunk_size` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '切片大小快照',
  `token_size` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Token 大小快照',
  `chunk_overlap` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '切片重叠大小快照',
  `embedding_model` VARCHAR(128) NOT NULL DEFAULT '' COMMENT '向量模型快照',
  `embedding_dim` INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '向量维度快照',
  `milvus_collection_name` VARCHAR(128) NOT NULL DEFAULT '' COMMENT 'Milvus 集合名称快照',
  `source_hash` CHAR(64) NOT NULL DEFAULT '' COMMENT '源内容哈希快照',
  `last_indexed_at` DATETIME(3) NULL DEFAULT NULL COMMENT '最近一次成功索引时间',
  `error_message` VARCHAR(2000) NOT NULL DEFAULT '' COMMENT '索引失败错误详情',
  `create_by` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '创建人账号',
  `update_by` VARCHAR(64) NOT NULL DEFAULT '' COMMENT '最后更新人账号',
  `created_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `updated_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  `is_deleted` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '逻辑删除标记：0 未删除，1 已删除',
  `deleted_at` DATETIME(3) NULL DEFAULT NULL COMMENT '逻辑删除时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_index_doc_ver_deleted` (`document_id`, `index_version`, `is_deleted`),
  KEY `idx_index_doc_status` (`document_id`, `index_status`, `is_deleted`),
  CONSTRAINT `fk_index_kb` FOREIGN KEY (`knowledge_base_id`) REFERENCES `knowledge_base` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `fk_index_doc` FOREIGN KEY (`document_id`) REFERENCES `knowledge_document` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='文档索引快照表';

SET FOREIGN_KEY_CHECKS = 1;

-- ============================================================
-- Example queries for backend integration
-- ============================================================

-- 1) Knowledge base list page
-- SELECT id, knowledge_name, display_name, status, updated_at
-- FROM knowledge_base
-- WHERE is_deleted = 0
-- ORDER BY id DESC
-- LIMIT :offset, :page_size;

-- 2) Document list under one knowledge base
-- SELECT id, knowledge_base_id, file_name, file_url, status, updated_at
-- FROM knowledge_document
-- WHERE knowledge_base_id = :knowledge_base_id
--   AND is_deleted = 0
-- ORDER BY id DESC
-- LIMIT :offset, :page_size;

-- 3) Import task list for one document
-- SELECT id, task_uuid, status, progress, error_message, started_at, finished_at, created_at
-- FROM knowledge_import_task
-- WHERE document_id = :document_id
--   AND is_deleted = 0
-- ORDER BY id DESC
-- LIMIT :offset, :page_size;

-- 4) Index snapshot details for one document
-- SELECT id, index_version, index_status, chunk_count, token_count,
--        chunk_strategy, chunk_size, token_size, chunk_overlap,
--        embedding_model, embedding_dim, milvus_collection_name,
--        source_hash, last_indexed_at, error_message
-- FROM knowledge_document_index
-- WHERE document_id = :document_id
--   AND is_deleted = 0
-- ORDER BY index_version DESC
-- LIMIT :offset, :page_size;
