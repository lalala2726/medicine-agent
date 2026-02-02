from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

import httpx

from app.core.request_context import get_authorization_header

logger = logging.getLogger(__name__)


class HttpClient:
    """
    基于 httpx 的轻量 HTTP 客户端封装。

    - 支持 GET/POST/PUT/DELETE
    - 支持 headers / query params / body
    - 提供统一的超时与基础 headers
    """

    def __init__(
            self,
            *,
            base_url: Optional[str] = None,
            headers: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = 30.0,
    ) -> None:
        if base_url is None:
            base_url = os.getenv("HTTP_BASE_URL", "http://localhost:8080")
        self._default_headers = dict(headers or {})
        self._client = httpx.AsyncClient(
            base_url=base_url or "",
            headers=self._default_headers,
            timeout=timeout,
        )

    def _build_headers(self, headers: Optional[Mapping[str, str]]) -> Mapping[str, str]:
        """
        合并请求头并注入当前请求的 Authorization。
        - 显式传入的 headers 优先
        - 若未显式传入 Authorization，则从请求上下文注入
        """
        merged = dict(self._default_headers)
        if headers:
            merged.update(headers)
        lower_keys = {key.lower() for key in merged}
        if "authorization" not in lower_keys:
            auth = get_authorization_header()
            if auth:
                merged["Authorization"] = auth
        return merged

    async def close(self) -> None:
        """关闭底层连接池。"""
        await self._client.aclose()

    async def __aenter__(self) -> "HttpClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def request(
            self,
            method: str,
            url: str,
            *,
            headers: Optional[Mapping[str, str]] = None,
            params: Optional[Mapping[str, Any]] = None,
            json: Optional[Any] = None,
            data: Optional[Mapping[str, Any]] = None,
            content: Optional[bytes] = None,
            timeout: Optional[float] = None,
    ) -> httpx.Response:
        """
        统一请求入口。

        Args:
            method: HTTP 方法
            url: 请求路径或完整 URL
            headers: 额外请求头（会覆盖同名默认头）
            params: Query 参数
            json: JSON body
            data: 表单 body
            content: 原始字节 body
            timeout: 本次请求超时
        """
        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=self._build_headers(headers),
                params=params,
                json=json,
                data=data,
                content=content,
                timeout=timeout,
            )
        except httpx.HTTPError:
            logger.exception(
                "HTTP request failed: method=%s url=%s",
                method,
                url,
            )
            raise

        if response.status_code >= 400:
            logger.warning(
                "HTTP response error: method=%s url=%s status=%s",
                method,
                url,
                response.status_code,
            )
            body = response.text if response.content else ""
            if body:
                snippet = body[:500]
                if len(body) > 500:
                    snippet = f"{snippet}...(truncated)"
                logger.warning("HTTP response body: %s", snippet)

        return response


    async def get(
            self,
            url: str,
            *,
            headers: Optional[Mapping[str, str]] = None,
            params: Optional[Mapping[str, Any]] = None,
            timeout: Optional[float] = None,
    ) -> httpx.Response:
        """发送 GET 请求。"""
        return await self.request(
            "GET",
            url,
            headers=headers,
            params=params,
            timeout=timeout,
        )


    async def post(
            self,
            url: str,
            *,
            headers: Optional[Mapping[str, str]] = None,
            params: Optional[Mapping[str, Any]] = None,
            json: Optional[Any] = None,
            data: Optional[Mapping[str, Any]] = None,
            content: Optional[bytes] = None,
            timeout: Optional[float] = None,
    ) -> httpx.Response:
        """发送 POST 请求。"""
        return await self.request(
            "POST",
            url,
            headers=headers,
            params=params,
            json=json,
            data=data,
            content=content,
            timeout=timeout,
        )

    async def put(
            self,
            url: str,
            *,
            headers: Optional[Mapping[str, str]] = None,
            params: Optional[Mapping[str, Any]] = None,
            json: Optional[Any] = None,
            data: Optional[Mapping[str, Any]] = None,
            content: Optional[bytes] = None,
            timeout: Optional[float] = None,
    ) -> httpx.Response:
        """发送 PUT 请求。"""
        return await self.request(
            "PUT",
            url,
            headers=headers,
            params=params,
            json=json,
            data=data,
            content=content,
            timeout=timeout,
        )

    async def delete(
            self,
            url: str,
            *,
            headers: Optional[Mapping[str, str]] = None,
            params: Optional[Mapping[str, Any]] = None,
            json: Optional[Any] = None,
            data: Optional[Mapping[str, Any]] = None,
            content: Optional[bytes] = None,
            timeout: Optional[float] = None,
    ) -> httpx.Response:
        """发送 DELETE 请求。"""
        return await self.request(
            "DELETE",
            url,
            headers=headers,
            params=params,
            json=json,
            data=data,
            content=content,
            timeout=timeout,
        )
