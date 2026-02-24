import os
from typing import Optional
from urllib.parse import urlparse

from pymilvus import MilvusClient

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException


def _build_milvus_uri() -> str:
    # 优先使用完整 URI，其次使用 host/port 组合
    uri = os.getenv("MILVUS_URI")
    if uri:
        return uri

    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    if host.startswith(("http://", "https://")):
        uri = host
    else:
        uri = f"http://{host}"

    parsed = urlparse(uri)
    if parsed.port is None:
        uri = f"{uri}:{port}"
    return uri


def get_milvus_client() -> MilvusClient:
    uri = _build_milvus_uri()
    user = os.getenv("MILVUS_USER") or os.getenv("MILVUS_USERNAME", "")
    password = os.getenv("MILVUS_PASSWORD", "")
    token = os.getenv("MILVUS_TOKEN", "")
    db_name = os.getenv("MILVUS_DB_NAME", "")
    timeout_value = os.getenv("MILVUS_TIMEOUT")
    timeout: Optional[float] = None
    if timeout_value:
        try:
            timeout = float(timeout_value)
        except ValueError as exc:
            raise ServiceException(
                code=ResponseCode.INTERNAL_ERROR,
                message="MILVUS_TIMEOUT 必须是数字",
            ) from exc
    return MilvusClient(
        uri=uri,
        user=user,
        password=password,
        token=token,
        db_name=db_name,
        timeout=timeout,
    )
