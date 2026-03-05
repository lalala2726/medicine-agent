from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.schemas.auth import AuthUser


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _mock_auth(monkeypatch) -> None:
    """注入固定认证上下文，避免依赖外部鉴权服务。"""

    async def _fake_fetch_current_user() -> AuthUser:
        return AuthUser(id=1, username="tester")

    monkeypatch.setattr(main_module, "verify_authorization", _fake_fetch_current_user)


def test_import_route_is_removed(monkeypatch) -> None:
    """验证纯 MQ 模式下 HTTP 导入提交接口已移除。"""
    _mock_auth(monkeypatch)
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

    assert response.status_code == 405
