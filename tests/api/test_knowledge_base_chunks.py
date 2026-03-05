from fastapi.testclient import TestClient

import app.main as main_module
from app.api.routes import knowledge_base as knowledge_base_route
from app.main import app
from app.schemas.auth import AuthUser


def _auth_headers() -> dict[str, str]:
    """
    功能描述:
        构造测试专用鉴权请求头。

    参数说明:
        无。

    返回值:
        dict[str, str]: 包含固定 Bearer Token 的请求头字典。

    异常说明:
        无。
    """
    return {"Authorization": "Bearer test-token"}


def _mock_auth(monkeypatch) -> None:
    """
    测试目的：为受鉴权保护接口注入固定认证上下文，避免测试依赖真实鉴权服务。
    预期结果：请求在携带测试 token 时可正常进入路由业务逻辑。
    """

    async def _fake_fetch_current_user() -> AuthUser:
        return AuthUser(id=1, username="tester")

    monkeypatch.setattr(
        main_module,
        "verify_authorization",
        _fake_fetch_current_user,
    )


def test_list_document_chunks_uses_page_size_default_50(monkeypatch) -> None:
    """
    测试目的：验证分页查询接口默认 page_size 为 50。
    预期结果：未传 page_size 时 service 收到 50，响应中也返回 50。
    """
    _mock_auth(monkeypatch)
    called: dict[str, int] = {}

    def _fake_list_knowledge_chunks(**kwargs):
        called["page_size"] = kwargs["page_size"]
        return [], 0

    monkeypatch.setattr(
        knowledge_base_route,
        "list_knowledge_chunks",
        _fake_list_knowledge_chunks,
    )
    client = TestClient(app)

    response = client.get(
        "/knowledge_base/document/chunks/list",
        headers=_auth_headers(),
        params={
            "knowledge_name": "demo_kb",
            "document_id": 1,
            "page": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert called["page_size"] == 50
    assert body["data"]["page_size"] == 50
    assert body["data"]["has_next"] is False


def test_list_document_chunks_rejects_page_size_over_100(monkeypatch) -> None:
    """
    测试目的：验证分页查询接口限制 page_size 最大值为 100。
    预期结果：当 page_size=101 时返回参数校验错误。
    """
    _mock_auth(monkeypatch)
    client = TestClient(app)

    response = client.get(
        "/knowledge_base/document/chunks/list",
        headers=_auth_headers(),
        params={
            "knowledge_name": "demo_kb",
            "document_id": 1,
            "page": 1,
            "page_size": 101,
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "Validation Failed"
    assert any(error["field"] == "page_size" for error in body["errors"])


def test_list_document_chunks_has_next_true_and_uses_chunk_index(monkeypatch) -> None:
    """
    测试目的：验证分页结果 has_next 计算正确，且 rows 使用 chunk_index 字段。
    预期结果：total=120/page=1/page_size=50 时 has_next=true。
    """
    _mock_auth(monkeypatch)

    def _fake_list_knowledge_chunks(**_kwargs):
        return [{"chunk_index": 1, "content": "A"}], 120

    monkeypatch.setattr(
        knowledge_base_route,
        "list_knowledge_chunks",
        _fake_list_knowledge_chunks,
    )
    client = TestClient(app)

    response = client.get(
        "/knowledge_base/document/chunks/list",
        headers=_auth_headers(),
        params={
            "knowledge_name": "demo_kb",
            "document_id": 1,
            "page": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["has_next"] is True
    assert body["data"]["rows"][0]["chunk_index"] == 1
    assert all(not key.endswith("_no") for key in body["data"]["rows"][0].keys())


def test_list_document_chunks_has_next_false_on_last_page(monkeypatch) -> None:
    """
    测试目的：验证最后一页 has_next 为 false。
    预期结果：total=120/page=3/page_size=50 时 has_next=false。
    """
    _mock_auth(monkeypatch)

    def _fake_list_knowledge_chunks(**_kwargs):
        return [{"chunk_index": 101, "content": "tail"}], 120

    monkeypatch.setattr(
        knowledge_base_route,
        "list_knowledge_chunks",
        _fake_list_knowledge_chunks,
    )
    client = TestClient(app)

    response = client.get(
        "/knowledge_base/document/chunks/list",
        headers=_auth_headers(),
        params={
            "knowledge_name": "demo_kb",
            "document_id": 1,
            "page": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["page_size"] == 50
    assert body["data"]["has_next"] is False
