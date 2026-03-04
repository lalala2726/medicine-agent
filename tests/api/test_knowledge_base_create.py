from fastapi.testclient import TestClient

import app.main as main_module
from app.api.routes import knowledge_base as knowledge_base_route
from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
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


def test_load_knowledge_base_passes_knowledge_name_and_returns_state(monkeypatch) -> None:
    """
    测试目的：验证启用接口透传 knowledge_name 并返回 load_state。
    预期结果：load_collection_state 被调用且响应 message 为“启用成功”。
    """
    _mock_auth(monkeypatch)
    called: dict[str, object] = {}

    def _fake_load_collection_state(knowledge_name: str) -> dict:
        called["knowledge_name"] = knowledge_name
        return {
            "knowledge_name": knowledge_name,
            "load_state": "Loaded",
        }

    monkeypatch.setattr(
        knowledge_base_route,
        "load_collection_state",
        _fake_load_collection_state,
    )
    client = TestClient(app)

    response = client.post(
        "/knowledge_base/load",
        headers=_auth_headers(),
        json={"knowledge_name": "demo_kb"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "启用成功"
    assert body["data"]["knowledge_name"] == "demo_kb"
    assert body["data"]["load_state"] == "Loaded"
    assert called["knowledge_name"] == "demo_kb"


def test_release_knowledge_base_passes_knowledge_name_and_returns_state(monkeypatch) -> None:
    """
    测试目的：验证关闭接口透传 knowledge_name 并返回 load_state。
    预期结果：release_collection_state 被调用且响应 message 为“关闭成功”。
    """
    _mock_auth(monkeypatch)
    called: dict[str, object] = {}

    def _fake_release_collection_state(knowledge_name: str) -> dict:
        called["knowledge_name"] = knowledge_name
        return {
            "knowledge_name": knowledge_name,
            "load_state": "NotLoad",
        }

    monkeypatch.setattr(
        knowledge_base_route,
        "release_collection_state",
        _fake_release_collection_state,
    )
    client = TestClient(app)

    response = client.post(
        "/knowledge_base/release",
        headers=_auth_headers(),
        json={"knowledge_name": "demo_kb"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "关闭成功"
    assert body["data"]["knowledge_name"] == "demo_kb"
    assert body["data"]["load_state"] == "NotLoad"
    assert called["knowledge_name"] == "demo_kb"


def test_load_knowledge_base_requires_knowledge_name(monkeypatch) -> None:
    """
    测试目的：验证启用接口将 knowledge_name 作为必填参数校验。
    预期结果：缺少 knowledge_name 时返回 400，且不触发 service。
    """
    _mock_auth(monkeypatch)

    def _unexpected_load_collection_state(*_args, **_kwargs):
        raise AssertionError("参数校验失败时不应触发 load_collection_state")

    monkeypatch.setattr(
        knowledge_base_route,
        "load_collection_state",
        _unexpected_load_collection_state,
    )
    client = TestClient(app)

    response = client.post(
        "/knowledge_base/load",
        headers=_auth_headers(),
        json={},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "Validation Failed"
    assert any(error["field"] == "knowledge_name" for error in body["errors"])


def test_load_knowledge_base_returns_not_found_when_service_raises(monkeypatch) -> None:
    """
    测试目的：验证启用接口可透传 service 层 NOT_FOUND 业务异常。
    预期结果：响应状态码与业务 code 均为 404。
    """
    _mock_auth(monkeypatch)

    def _raise_not_found(_knowledge_name: str) -> dict:
        raise ServiceException(
            code=ResponseCode.NOT_FOUND,
            message="知识库不存在",
        )

    monkeypatch.setattr(
        knowledge_base_route,
        "load_collection_state",
        _raise_not_found,
    )
    client = TestClient(app)

    response = client.post(
        "/knowledge_base/load",
        headers=_auth_headers(),
        json={"knowledge_name": "demo_kb"},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == 404
    assert body["message"] == "知识库不存在"
