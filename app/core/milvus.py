import os
from typing import Optional

from pymilvus import MilvusClient

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException


def get_milvus_client() -> MilvusClient:
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    user = os.getenv("MILVUS_USER", "")
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
