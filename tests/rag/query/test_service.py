from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.codes import ResponseCode
from app.core.config_sync import AgentConfigSnapshot
from app.core.exception.exceptions import ServiceException
from app.rag.query import KnowledgeSearchHit
from app.rag.query import service as service_module

_KNOWLEDGE_ENABLED_UNSET = object()


class _FakeMilvusClient:
    def __init__(
            self,
            *,
            existing_collections: set[str],
            results_by_collection: dict[str, list[dict[str, Any]]],
    ) -> None:
        self._existing_collections = existing_collections
        self._results_by_collection = results_by_collection
        self.has_collection_calls: list[str] = []
        self.search_calls: list[dict[str, Any]] = []

    def has_collection(self, collection_name: str) -> bool:
        self.has_collection_calls.append(collection_name)
        return collection_name in self._existing_collections

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.search_calls.append(kwargs)
        collection_name = kwargs["collection_name"]
        results = self._results_by_collection.get(collection_name, [])
        return [results]


class _FakeRankingModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.invoke_calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.invoke_calls.append(messages)
        return SimpleNamespace(content=self.content)


def _build_snapshot(
        *,
        provider_type: str = "aliyun",
        knowledge_names: list[str] | None = None,
        embedding_model: str = "text-embedding-v4",
        embedding_dim: int = 1024,
        ranking_enabled: bool = False,
        ranking_model: str | None = None,
        top_k: int | None = 8,
        knowledge_enabled: bool | None | object = _KNOWLEDGE_ENABLED_UNSET,
) -> AgentConfigSnapshot:
    knowledge_base: dict[str, Any] = {
        "knowledgeNames": knowledge_names or ["common_medicine_kb", "otc_guide_kb"],
        "embeddingDim": embedding_dim,
        "embeddingModel": embedding_model,
        "rankingEnabled": ranking_enabled,
        "rankingModel": ranking_model,
        "topK": top_k,
    }
    if knowledge_enabled is not _KNOWLEDGE_ENABLED_UNSET:
        knowledge_base["enabled"] = knowledge_enabled

    return AgentConfigSnapshot.model_validate(
        {
            "schemaVersion": 4,
            "updatedAt": "2026-03-14T10:30:00+08:00",
            "updatedBy": "admin",
            "llm": {
                "providerType": provider_type,
                "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "apiKey": "sk-runtime",
            },
            "agentConfigs": {
                "knowledgeBase": knowledge_base,
            },
        },
    )


def _build_runtime_config(
        *,
        provider_type: str = "aliyun",
        knowledge_names: list[str] | None = None,
        ranking_enabled: bool = False,
        ranking_model_name: str | None = None,
        configured_top_k: int | None = 8,
) -> service_module._KnowledgeSearchRuntimeConfig:
    return service_module._KnowledgeSearchRuntimeConfig(
        provider_type=provider_type,
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_api_key="sk-runtime",
        knowledge_names=knowledge_names or ["common_medicine_kb", "otc_guide_kb"],
        embedding_model_name="text-embedding-v4",
        embedding_dim=1024,
        ranking_enabled=ranking_enabled,
        ranking_model_name=ranking_model_name,
        configured_top_k=configured_top_k,
    )


def test_build_rag_embedding_client_uses_redis_runtime_config(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        service_module,
        "create_embedding_model",
        lambda **kwargs: captured.update(kwargs) or "embedding-client",
    )

    result = service_module._build_rag_embedding_client(
        runtime_config=_build_runtime_config(provider_type="openai"),
    )

    assert result == "embedding-client"
    assert captured == {
        "provider": "openai",
        "model": "text-embedding-v4",
        "api_key": "sk-runtime",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "dimensions": 1024,
    }


def test_log_rag_runtime_config_once_only_logs_once(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_logs: list[tuple[Any, ...]] = []
    monkeypatch.setattr(service_module, "_RAG_RUNTIME_CONFIG_LOGGED", False)
    monkeypatch.setattr(service_module.logger, "info", lambda *args: captured_logs.append(args))

    runtime_config = _build_runtime_config(
        ranking_enabled=True,
        ranking_model_name="gpt-4.1-mini",
        configured_top_k=6,
    )
    connection_args = {"uri": "http://milvus:19530", "db_name": "medicine"}
    service_module._log_rag_runtime_config_once(
        runtime_config=runtime_config,
        connection_args=connection_args,
    )
    service_module._log_rag_runtime_config_once(
        runtime_config=runtime_config,
        connection_args=connection_args,
    )

    assert len(captured_logs) == 1
    assert "RAG 查询配置已生效" in captured_logs[0][0]
    assert captured_logs[0][2] == "text-embedding-v4"
    assert captured_logs[0][7] == ["common_medicine_kb", "otc_guide_kb"]
    assert captured_logs[0][8] is True
    assert captured_logs[0][9] == "gpt-4.1-mini"


def test_resolve_runtime_config_rejects_more_than_10_knowledge_names(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_module.runtime_module,
        "get_current_agent_config_snapshot",
        lambda: _build_snapshot(
            knowledge_names=[f"kb_{index}" for index in range(11)],
            ranking_enabled=True,
            ranking_model="gpt-4.1-mini",
        ),
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module.runtime_module.resolve_runtime_config()

    assert exc_info.value.code == ResponseCode.SERVICE_UNAVAILABLE.code
    assert "不能超过 10 个" in exc_info.value.message


def test_resolve_runtime_config_rejects_disabled_knowledge_base(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_module.runtime_module,
        "get_current_agent_config_snapshot",
        lambda: _build_snapshot(knowledge_enabled=False),
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module.runtime_module.resolve_runtime_config()

    assert exc_info.value.code == ResponseCode.SERVICE_UNAVAILABLE.code
    assert exc_info.value.message == "知识库检索未启用"


def test_resolve_runtime_config_accepts_legacy_knowledge_base_without_enabled(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_module.runtime_module,
        "get_current_agent_config_snapshot",
        _build_snapshot,
    )

    runtime_config = service_module.runtime_module.resolve_runtime_config()

    assert runtime_config.knowledge_names == ["common_medicine_kb", "otc_guide_kb"]
    assert runtime_config.embedding_model_name == "text-embedding-v4"
    assert runtime_config.configured_top_k == 8


def test_query_knowledge_by_raw_question_aggregates_multi_collections(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = _build_runtime_config(knowledge_names=["common_medicine_kb", "otc_guide_kb"])
    fake_client = _FakeMilvusClient(
        existing_collections={"common_medicine_kb", "otc_guide_kb"},
        results_by_collection={
            "common_medicine_kb": [
                {
                    "distance": 0.81,
                    "entity": {
                        "content": "基础药品说明",
                        "document_id": 1,
                        "chunk_index": 1,
                        "char_count": 6,
                    },
                },
            ],
            "otc_guide_kb": [
                {
                    "distance": 0.93,
                    "entity": {
                        "content": "OTC 指南内容",
                        "document_id": 2,
                        "chunk_index": 3,
                        "char_count": 8,
                    },
                },
            ],
        },
    )
    monkeypatch.setattr(service_module.runtime_module, "resolve_runtime_config", lambda: runtime_config)
    monkeypatch.setattr(service_module, "_build_rag_milvus_client", lambda **_: fake_client)
    monkeypatch.setattr(
        service_module,
        "_build_rag_embedding_client",
        lambda **_: type("EmbeddingClient", (), {"embed_query": lambda self, text: [0.1, 0.2]})(),
    )

    hits = service_module.query_knowledge_by_raw_question(question="感冒药怎么吃", top_k=2)

    assert hits == [
        KnowledgeSearchHit(
            knowledge_name="otc_guide_kb",
            content="OTC 指南内容",
            score=0.93,
            document_id=2,
            chunk_index=3,
            char_count=8,
        ),
        KnowledgeSearchHit(
            knowledge_name="common_medicine_kb",
            content="基础药品说明",
            score=0.81,
            document_id=1,
            chunk_index=1,
            char_count=6,
        ),
    ]


def test_query_knowledge_by_raw_question_skips_missing_collection(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = _build_runtime_config(knowledge_names=["missing_kb", "otc_guide_kb"])
    fake_client = _FakeMilvusClient(
        existing_collections={"otc_guide_kb"},
        results_by_collection={
            "otc_guide_kb": [
                {
                    "distance": 0.93,
                    "entity": {
                        "content": "OTC 指南内容",
                    },
                },
            ],
        },
    )
    monkeypatch.setattr(service_module.runtime_module, "resolve_runtime_config", lambda: runtime_config)
    monkeypatch.setattr(service_module, "_build_rag_milvus_client", lambda **_: fake_client)
    monkeypatch.setattr(
        service_module,
        "_build_rag_embedding_client",
        lambda **_: type("EmbeddingClient", (), {"embed_query": lambda self, text: [0.1, 0.2]})(),
    )

    hits = service_module.query_knowledge_by_raw_question(question="感冒药怎么吃", top_k=2)

    assert [item.knowledge_name for item in hits] == ["otc_guide_kb"]
    assert fake_client.has_collection_calls == ["missing_kb", "otc_guide_kb"]


def test_query_knowledge_by_raw_question_raises_when_all_collections_missing(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = _build_runtime_config(knowledge_names=["missing_a", "missing_b"])
    fake_client = _FakeMilvusClient(
        existing_collections=set(),
        results_by_collection={},
    )
    monkeypatch.setattr(service_module.runtime_module, "resolve_runtime_config", lambda: runtime_config)
    monkeypatch.setattr(service_module, "_build_rag_milvus_client", lambda **_: fake_client)
    monkeypatch.setattr(
        service_module,
        "_build_rag_embedding_client",
        lambda **_: type("EmbeddingClient", (), {"embed_query": lambda self, text: [0.1, 0.2]})(),
    )

    with pytest.raises(ServiceException) as exc_info:
        service_module.query_knowledge_by_raw_question(question="感冒药怎么吃", top_k=2)

    assert exc_info.value.code == ResponseCode.NOT_FOUND.code
    assert "知识库集合不存在" in exc_info.value.message


def test_query_knowledge_by_raw_question_prefers_explicit_top_k_over_redis(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        service_module.runtime_module,
        "resolve_runtime_config",
        lambda: _build_runtime_config(configured_top_k=8),
    )
    monkeypatch.setattr(
        service_module,
        "_search_knowledge_hits",
        lambda *, question, final_top_k, runtime_config, ranking_enabled: (
            captured.update({"question": question, "final_top_k": final_top_k, "ranking_enabled": ranking_enabled}),
            [],
        )[-1],
    )

    hits = service_module.query_knowledge_by_raw_question(question="感冒药", top_k=3)

    assert hits == []
    assert captured["final_top_k"] == 3
    assert captured["ranking_enabled"] is False


def test_query_knowledge_by_raw_question_uses_redis_top_k_when_explicit_missing(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        service_module.runtime_module,
        "get_current_agent_config_snapshot",
        lambda: _build_snapshot(top_k=6),
    )
    monkeypatch.setattr(
        service_module.runtime_module,
        "resolve_runtime_config",
        lambda: _build_runtime_config(configured_top_k=6),
    )
    monkeypatch.setattr(
        service_module,
        "_search_knowledge_hits",
        lambda *, question, final_top_k, runtime_config, ranking_enabled: (
            captured.update({"final_top_k": final_top_k}),
            [],
        )[-1],
    )

    hits = service_module.query_knowledge_by_raw_question(question="感冒药", top_k=None)

    assert hits == []
    assert captured["final_top_k"] == 6


def test_query_knowledge_by_raw_question_falls_back_to_default_top_k(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        service_module.runtime_module,
        "resolve_runtime_config",
        lambda: _build_runtime_config(configured_top_k=None),
    )
    monkeypatch.setattr(
        service_module,
        "_search_knowledge_hits",
        lambda *, question, final_top_k, runtime_config, ranking_enabled: (
            captured.update({"final_top_k": final_top_k}),
            [],
        )[-1],
    )

    hits = service_module.query_knowledge_by_raw_question(question="感冒药", top_k=None)

    assert hits == []
    assert captured["final_top_k"] == service_module.runtime_module.RAG_DEFAULT_FINAL_TOP_K


def test_search_knowledge_hits_evenly_distributes_recall_without_ranking(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_names = [f"kb_{index}" for index in range(10)]
    runtime_config = _build_runtime_config(
        knowledge_names=knowledge_names,
        ranking_enabled=False,
        configured_top_k=10,
    )
    fake_client = _FakeMilvusClient(
        existing_collections=set(knowledge_names),
        results_by_collection={
            knowledge_name: [
                {
                    "distance": 1.0 - index * 0.01,
                    "entity": {
                        "content": f"{knowledge_name}-content",
                    },
                }
            ]
            for index, knowledge_name in enumerate(knowledge_names)
        },
    )
    monkeypatch.setattr(service_module, "_build_rag_milvus_client", lambda **_: fake_client)
    monkeypatch.setattr(
        service_module,
        "_build_rag_embedding_client",
        lambda **_: type("EmbeddingClient", (), {"embed_query": lambda self, text: [0.1, 0.2]})(),
    )

    hits = service_module._search_knowledge_hits(
        question="毕业论文",
        final_top_k=10,
        runtime_config=runtime_config,
        ranking_enabled=False,
    )

    assert len(hits) == 10
    assert all(call["limit"] == 1 for call in fake_client.search_calls)


def test_search_knowledge_hits_uses_expanded_candidate_pool_when_ranking_enabled(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_names = ["kb_a", "kb_b"]
    runtime_config = _build_runtime_config(
        knowledge_names=knowledge_names,
        ranking_enabled=True,
        ranking_model_name="gpt-4.1-mini",
        configured_top_k=4,
    )
    fake_client = _FakeMilvusClient(
        existing_collections=set(knowledge_names),
        results_by_collection={
            "kb_a": [{"distance": 0.9, "entity": {"content": "A1"}}],
            "kb_b": [{"distance": 0.8, "entity": {"content": "B1"}}],
        },
    )
    monkeypatch.setattr(service_module, "_build_rag_milvus_client", lambda **_: fake_client)
    monkeypatch.setattr(
        service_module,
        "_build_rag_embedding_client",
        lambda **_: type("EmbeddingClient", (), {"embed_query": lambda self, text: [0.1, 0.2]})(),
    )

    service_module._search_knowledge_hits(
        question="毕业论文",
        final_top_k=4,
        runtime_config=runtime_config,
        ranking_enabled=True,
    )

    assert all(call["limit"] == 6 for call in fake_client.search_calls)


def test_rank_hits_with_chat_model_uses_json_serial_numbers_and_reorders_hits(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [
        KnowledgeSearchHit(knowledge_name="kb_a", content="内容A", score=0.9),
        KnowledgeSearchHit(knowledge_name="kb_b", content="内容B", score=0.8),
        KnowledgeSearchHit(knowledge_name="kb_c", content="内容C", score=0.7),
    ]
    fake_model = _FakeRankingModel('{"top_serial_numbers":[2,1]}')
    monkeypatch.setattr(service_module, "_build_ranking_chat_model", lambda **_: fake_model)

    ranked_hits = service_module._rank_hits_with_chat_model(
        query="什么是毕业论文数据字段定义",
        hits=hits,
        runtime_config=_build_runtime_config(ranking_enabled=True, ranking_model_name="gpt-4.1-mini"),
        final_top_k=2,
    )

    assert [hit.knowledge_name for hit in ranked_hits] == ["kb_b", "kb_a"]
    request_payload = json.loads(fake_model.invoke_calls[0][1].content)
    assert request_payload["query"] == "什么是毕业论文数据字段定义"
    assert request_payload["top_n"] == 2
    assert request_payload["documents"][0]["serial_no"] == 1
    assert request_payload["documents"][1]["knowledge_name"] == "kb_b"


def test_rank_hits_with_chat_model_ignores_invalid_serials_and_backfills(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [
        KnowledgeSearchHit(knowledge_name="kb_a", content="内容A", score=0.9),
        KnowledgeSearchHit(knowledge_name="kb_b", content="内容B", score=0.8),
        KnowledgeSearchHit(knowledge_name="kb_c", content="内容C", score=0.7),
    ]
    fake_model = _FakeRankingModel('{"top_serial_numbers":[2,"x",2,99]}')
    monkeypatch.setattr(service_module, "_build_ranking_chat_model", lambda **_: fake_model)

    ranked_hits = service_module._rank_hits_with_chat_model(
        query="问题",
        hits=hits,
        runtime_config=_build_runtime_config(ranking_enabled=True, ranking_model_name="gpt-4.1-mini"),
        final_top_k=3,
    )

    assert [hit.knowledge_name for hit in ranked_hits] == ["kb_b", "kb_a", "kb_c"]


def test_query_knowledge_by_raw_question_falls_back_to_vector_when_ranking_fails(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = _build_runtime_config(
        ranking_enabled=True,
        ranking_model_name="gpt-4.1-mini",
        configured_top_k=3,
    )
    captured_calls: list[bool] = []
    ranking_candidates = [
        KnowledgeSearchHit(knowledge_name="kb_a", content="A", score=0.99),
        KnowledgeSearchHit(knowledge_name="kb_b", content="B", score=0.98),
        KnowledgeSearchHit(knowledge_name="kb_c", content="C", score=0.97),
    ]
    fallback_hits = [
        KnowledgeSearchHit(knowledge_name="kb_a", content="A", score=0.91),
        KnowledgeSearchHit(knowledge_name="kb_b", content="B", score=0.90),
    ]
    monkeypatch.setattr(service_module.runtime_module, "resolve_runtime_config", lambda: runtime_config)
    monkeypatch.setattr(
        service_module,
        "_search_knowledge_hits",
        lambda *, question, final_top_k, runtime_config, ranking_enabled: (
            captured_calls.append(ranking_enabled),
            ranking_candidates if ranking_enabled else fallback_hits,
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "_rank_hits_with_chat_model",
        lambda **_: (_ for _ in ()).throw(ValueError("bad ranking json")),
    )

    hits = service_module.query_knowledge_by_raw_question(question="毕业论文字段", top_k=3)

    assert captured_calls == [True, False]
    assert hits == fallback_hits


def test_query_knowledge_by_rewritten_question_uses_original_question_for_ranking(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = _build_runtime_config(
        ranking_enabled=True,
        ranking_model_name="gpt-4.1-mini",
        configured_top_k=2,
    )
    captured: dict[str, Any] = {}
    vector_hits = [
        KnowledgeSearchHit(knowledge_name="kb_a", content="A", score=0.9),
        KnowledgeSearchHit(knowledge_name="kb_b", content="B", score=0.8),
    ]
    monkeypatch.setattr(service_module, "_rewrite_question_for_knowledge_search", lambda question: "改写后的问题")
    monkeypatch.setattr(service_module.runtime_module, "resolve_runtime_config", lambda: runtime_config)
    monkeypatch.setattr(
        service_module,
        "_search_knowledge_hits",
        lambda *, question, final_top_k, runtime_config, ranking_enabled: (
            captured.update({"vector_question": question}),
            vector_hits,
        )[-1],
    )
    monkeypatch.setattr(
        service_module,
        "_rank_hits_with_chat_model",
        lambda *, query, hits, runtime_config, final_top_k: (
            captured.update({"ranking_question": query}),
            hits,
        )[-1],
    )

    hits = service_module.query_knowledge_by_rewritten_question(
        question="  原始问题  ",
        top_k=2,
    )

    assert hits == vector_hits
    assert captured == {
        "vector_question": "改写后的问题",
        "ranking_question": "原始问题",
    }


def test_format_knowledge_search_hits_includes_knowledge_name_and_respects_budget() -> None:
    oversized_content = "a" * (service_module.RAG_MAX_CONTEXT_CHARS + 100)
    rendered = service_module.format_knowledge_search_hits(
        [
            KnowledgeSearchHit(
                knowledge_name="common_medicine_kb",
                content=oversized_content,
                score=0.95,
                document_id=11,
                chunk_index=4,
                char_count=len(oversized_content),
            )
        ]
    )

    assert rendered.startswith("已检索到以下知识片段：")
    assert "knowledge_name=common_medicine_kb" in rendered
    assert "document_id=11" in rendered
    assert "..." in rendered
    assert len(rendered) <= service_module.RAG_MAX_CONTEXT_CHARS + 200
