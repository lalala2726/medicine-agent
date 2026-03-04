from fastapi.testclient import TestClient

import app.main as main_module
from app.api.routes import knowledge_base as knowledge_base_route
from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.main import app
from app.schemas.auth import AuthUser


def _auth_headers() -> dict[str, str]:
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


def test_import_route_submits_message_to_mq(monkeypatch) -> None:
    """
    测试目的：验证导入接口不会直接执行导入处理，而是调用 MQ 提交函数。
    预期结果：submit_import_to_queue 被调用且响应返回“异步队列处理中”文案。
    """
    _mock_auth(monkeypatch)
    called: dict = {}

    async def _fake_submit(**kwargs):
        called.update(kwargs)
        return {"accepted_count": 1, "task_uuids": ["task-1"]}

    monkeypatch.setattr(knowledge_base_route, "submit_import_to_queue", _fake_submit)
    client = TestClient(app)

    response = client.post(
        "/knowledge_base/document/import",
        headers=_auth_headers(),
        json={
            "knowledge_name": "demo",
            "document_id": 100,
            "file_urls": ["https://example.com/demo.txt"],
            "embedding_model": "text-embedding-v4",
            "chunk_strategy": "character",
            "chunk_size": 500,
            "token_size": 100,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "操作成功"
    assert body["data"] == "已接收导入请求，正在异步队列处理中～"
    assert called["knowledge_name"] == "demo"
    assert called["document_id"] == 100
    assert called["file_url"] == ["https://example.com/demo.txt"]
    assert called["embedding_model"] == "text-embedding-v4"


def test_import_route_keeps_file_urls_and_submits_per_request(monkeypatch) -> None:
    """
    测试目的：验证导入接口保持 file_urls 入参语义并原样传递给 MQ 提交服务。
    预期结果：submit_import_to_queue 接收到完整 file_urls 数组。
    """
    _mock_auth(monkeypatch)
    called: dict = {}

    async def _fake_submit(**kwargs):
        called.update(kwargs)
        return {"accepted_count": 2, "task_uuids": ["task-1", "task-2"]}

    monkeypatch.setattr(knowledge_base_route, "submit_import_to_queue", _fake_submit)
    client = TestClient(app)

    response = client.post(
        "/knowledge_base/document/import",
        headers=_auth_headers(),
        json={
            "knowledge_name": "demo",
            "document_id": 101,
            "file_urls": [
                "https://example.com/a.txt",
                "https://example.com/b.txt",
            ],
            "embedding_model": "text-embedding-v4",
            "chunk_strategy": "character",
            "chunk_size": 500,
            "token_size": 100,
        },
    )

    assert response.status_code == 200
    assert called["file_url"] == [
        "https://example.com/a.txt",
        "https://example.com/b.txt",
    ]


def test_import_route_returns_error_when_mq_submit_fails(monkeypatch) -> None:
    """
    测试目的：验证 MQ 提交失败时接口返回错误，不降级为同步执行。
    预期结果：响应状态码为 400，错误消息为模拟的 MQ 提交失败信息。
    """
    _mock_auth(monkeypatch)

    async def _fake_submit(**_kwargs):
        raise ServiceException(
            code=ResponseCode.OPERATION_FAILED,
            message="MQ 提交失败",
        )

    monkeypatch.setattr(knowledge_base_route, "submit_import_to_queue", _fake_submit)
    client = TestClient(app)

    response = client.post(
        "/knowledge_base/document/import",
        headers=_auth_headers(),
        json={
            "knowledge_name": "demo",
            "document_id": 102,
            "file_urls": ["https://example.com/a.txt"],
            "embedding_model": "text-embedding-v4",
            "chunk_strategy": "character",
            "chunk_size": 500,
            "token_size": 100,
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "MQ 提交失败"


def test_import_route_rejects_missing_embedding_model(monkeypatch) -> None:
    """
    测试目的：验证导入接口将 embedding_model 作为必填参数校验。
    预期结果：缺少 embedding_model 时返回 400，且不触发 MQ 提交。
    """
    _mock_auth(monkeypatch)

    async def _unexpected_submit(**_kwargs):
        raise AssertionError("参数校验失败时不应触发 MQ 提交")

    monkeypatch.setattr(knowledge_base_route, "submit_import_to_queue", _unexpected_submit)
    client = TestClient(app)

    response = client.post(
        "/knowledge_base/document/import",
        headers=_auth_headers(),
        json={
            "knowledge_name": "demo",
            "document_id": 103,
            "file_urls": ["https://example.com/a.txt"],
            "chunk_strategy": "character",
            "chunk_size": 500,
            "token_size": 100,
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 400
    assert body["message"] == "Validation Failed"
    assert any(error["field"] == "embedding_model" for error in body["errors"])
