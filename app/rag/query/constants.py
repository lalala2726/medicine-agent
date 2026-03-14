from __future__ import annotations

from app.repositories import vector_repository
from app.utils.prompt_utils import load_prompt

#: 知识库切片检索时只命中启用状态数据。
RAG_FILTER_EXPR = "status == 0"
#: 复用项目当前 Milvus 向量索引的默认检索参数。
RAG_SEARCH_PARAMS = {"metric_type": vector_repository.DEFAULT_VECTOR_METRIC_TYPE, "params": {}}
#: 知识检索返回给查询层的标准输出字段。
RAG_OUTPUT_FIELDS = ["document_id", "chunk_index", "char_count", "content"]
#: 当显式参数与 Redis 都未提供时的默认最终返回条数。
RAG_DEFAULT_FINAL_TOP_K = 10
#: 最终返回条数允许的最大值，避免上下文被无限放大。
RAG_MAX_FINAL_TOP_K = 100
#: 输出给 Agent 的知识上下文最大字符预算。
RAG_MAX_CONTEXT_CHARS = 12000
#: 单次最多允许同时查询的知识库数量。
RAG_MAX_KNOWLEDGE_NAMES = 10
#: 启用排序时的候选池最大规模。
RAG_MAX_CANDIDATE_POOL = 100
#: 排序模型的固定最大输出 token。
RAG_RANKING_MAX_TOKENS = 512
#: 要求底层聊天模型尽量返回合法 JSON。
RAG_RANKING_RESPONSE_FORMAT = {"response_format": {"type": "json_object"}}
#: 检索问题改写链路使用的系统提示词。
RAG_REWRITE_PROMPT = load_prompt("_system/rewrite_rag_query.md").strip()
#: 知识片段排序链路使用的系统提示词。
RAG_RANKING_PROMPT = """
    你是知识库结果排序器。你的任务是根据用户问题，从候选文档中选出最能直接回答问题的前 top_n 个片段。

    必须遵守：
    1. 仅输出一个 JSON 对象。
    2. JSON 结构固定为 {"top_serial_numbers":[整数序号,...]}。
    3. 只能返回候选 documents 中已经出现过的 serial_no。
    4. 只返回前 top_n 个序号，不要解释，不要附加 markdown，不要输出其他字段。
    5. 优先选择最能回答问题的片段，而不是仅仅主题相近的片段。
    6. 如果候选里没有足够合适的片段，也只返回你认为最相关的序号。
""".strip()
