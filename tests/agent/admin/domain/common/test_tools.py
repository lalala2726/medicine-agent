from __future__ import annotations

import pytest

from app.agent.admin.domain.common import tools as tools_module
from app.rag import KnowledgeSearchHit


def test_search_knowledge_context_invokes_rewritten_query_and_formats_result(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        tools_module,
        "query_knowledge_by_rewritten_question",
        lambda *, question, top_k: (
            captured.update({"question": question, "top_k": top_k}),
            [
                KnowledgeSearchHit(
                    knowledge_name="common_medicine_kb",
                    content="连花清瘟用于感冒相关症状缓解。",
                    score=0.93,
                    document_id=11,
                    chunk_index=4,
                    char_count=16,
                )
            ],
        )[-1],
    )

    result = tools_module.search_knowledge_context.invoke(
        {"query": "  连花清瘟说明书  "}
    )

    assert captured == {
        "question": "连花清瘟说明书",
        "top_k": None,
    }
    assert "已检索到以下知识片段：" in result
    assert "knowledge_name=common_medicine_kb" in result
    assert "document_id=11" in result
    assert "chunk_index=4" in result
    assert "连花清瘟用于感冒相关症状缓解。" in result


def test_search_knowledge_context_uses_redis_top_k_when_not_explicit(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        tools_module,
        "query_knowledge_by_rewritten_question",
        lambda *, question, top_k: (
            captured.update({"question": question, "top_k": top_k}),
            [],
        )[-1],
    )

    result = tools_module.search_knowledge_context.invoke({"query": "  连花清瘟说明书  "})

    assert result == "未检索到相关知识。"
    assert captured == {
        "question": "连花清瘟说明书",
        "top_k": None,
    }
