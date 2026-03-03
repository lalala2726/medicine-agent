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


def test_create_knowledge_base_requires_embedding_dim(monkeypatch) -> None:
    """
    测试目的：验证创建知识库接口将 embedding_dim 作为必填参数校验。
    预期结果：缺少 embedding_dim 时返回 400，错误字段包含 embedding_dim。
    """
    _mock_auth(monkeypatch)

    def _unexpected_create(*_args, **_kwargs):
        raise AssertionError("参数校验失败时不应触发 create_collection")

    monkeypatch.setattr(knowledge_base_route, "create_collection", _unexpected_create)
    client = TestClient(app)

    response = client.post(
        "/knowledge_base",
        headers=_auth_headers(),
        json={
            "knowledge_name": "demo_kb",
            "description": "demo",
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "Validation Failed"
    assert any(error["field"] == "embedding_dim" for error in body["errors"])


def test_create_knowledge_base_passes_required_fields_to_service(monkeypatch) -> None:
    """
    测试目的：验证创建知识库接口会将核心参数透传给 service 层建库函数。
    预期结果：create_collection 被调用且参数包含 knowledge_name、embedding_dim、description。
    """
    _mock_auth(monkeypatch)
    called: dict[str, object] = {}

    def _fake_create_collection(
            knowledge_name: str,
            embedding_dim: int,
            description: str,
    ) -> None:
        called["knowledge_name"] = knowledge_name
        called["embedding_dim"] = embedding_dim
        called["description"] = description

    monkeypatch.setattr(knowledge_base_route, "create_collection", _fake_create_collection)
    client = TestClient(app)

    response = client.post(
        "/knowledge_base",
        headers=_auth_headers(),
        json={
            "knowledge_name": "demo_kb",
            "embedding_dim": 1024,
            "description": "demo",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "创建成功"
    assert body["data"]["knowledge_name"] == "demo_kb"
    assert called["knowledge_name"] == "demo_kb"
    assert called["embedding_dim"] == 1024
    assert called["description"] == "demo"
