from __future__ import annotations

import json as json_lib
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import httpx
import yaml
from dotenv import load_dotenv
from loguru import logger

from app.core.security.auth_context import get_authorization_header


class HttpClient:
    """
    基于 httpx 的轻量 HTTP 客户端封装。

    - 支持 GET/POST/PUT/DELETE
    - 支持 headers / query params / body
    - 提供统一的超时与基础 headers
    """

    _dotenv_checked = False

    def __init__(
            self,
            *,
            base_url: Optional[str] = None,
            headers: Optional[Mapping[str, str]] = None,
            timeout: Optional[float] = 30.0,
            agent_key: Optional[str] = None,
    ) -> None:
        self._ensure_env_loaded()
        if base_url is None:
            base_url = os.getenv("HTTP_BASE_URL", "http://localhost:8083")
        self._default_log_enabled = self._parse_bool(
            os.getenv("HTTP_CLIENT_LOG_ENABLED")
        )
        self._agent_key = agent_key or os.getenv("X_AGENT_KEY", "")
        self._default_headers = dict(headers or {})
        self._client = httpx.AsyncClient(
            base_url=base_url or "",
            headers=self._default_headers,
            timeout=timeout,
        )

    @staticmethod
    def _parse_bool(value: Optional[str | bool]) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _ensure_env_loaded(cls) -> None:
        """
        兜底加载项目根目录 .env。
        避免在非 FastAPI 入口（例如脚本、LangGraph 本地运行）下没有提前 load_dotenv 导致配置不生效。
        """
        if cls._dotenv_checked:
            return
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
        cls._dotenv_checked = True

    def _is_log_enabled(self) -> bool:
        value = os.getenv("HTTP_CLIENT_LOG_ENABLED")
        if value is None:
            return self._default_log_enabled
        return self._parse_bool(value)

    @staticmethod
    def _redact_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
        sensitive_keys = {
            "authorization",
            "proxy-authorization",
            "x-api-key",
            "x-agent-key",
        }
        return {
            key: "***" if key.lower() in sensitive_keys else value
            for key, value in headers.items()
        }

    @staticmethod
    def _truncate(value: Any, limit: int = 1000) -> str:
        if value is None:
            return ""
        try:
            if isinstance(value, (dict, list, tuple)):
                text = json_lib.dumps(value)
            else:
                text = str(value)
        except Exception:
            text = repr(value)
        if len(text) > limit:
            return f"{text[:limit]}...(truncated)"
        return text

    def _build_headers(self, headers: Optional[Mapping[str, str]]) -> Mapping[str, str]:
        """
        合并请求头并注入当前请求的 Authorization。
        - 显式传入的 headers 优先
        - 若未显式传入 Authorization，则从请求上下文注入
        - 自动添加 X-Agent-Key、X-Agent-Timestamp、X-Agent-Nonce
        """
        merged = dict(self._default_headers)
        if headers:
            merged.update(headers)
        lower_keys = {key.lower() for key in merged}
        if "authorization" not in lower_keys:
            auth = get_authorization_header()
            if auth:
                merged["Authorization"] = auth

        # 添加 Agent 相关请求头
        if self._agent_key:
            merged["X-Agent-Key"] = self._agent_key
        merged["X-Agent-Timestamp"] = str(int(time.time()))
        merged["X-Agent-Nonce"] = str(uuid.uuid4())

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
            response_format: Literal["raw", "json", "yaml"] = "raw",
            include_envelope: bool = False,
    ) -> Any:
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
            response_format: 返回格式。`raw` 返回 `httpx.Response`，
                `json` 返回 Python 对象，`yaml` 返回 YAML 字符串。
            include_envelope: 当 `response_format` 为 `json/yaml` 时，
                是否保留 `code/message/data/timestamp` 包裹结构。
        """
        log_enabled = self._is_log_enabled()
        start = time.monotonic()
        try:
            built_headers = self._build_headers(headers)
        except Exception as exc:
            if log_enabled:
                logger.warning(
                    "HTTP request blocked before send: method={} url={} reason={}",
                    method,
                    url,
                    exc,
                )
            raise

        if log_enabled:
            logger.info(
                "HTTP request: method={} url={} headers={} params={} json={} data={} content_length={}",
                method,
                url,
                self._redact_headers(built_headers),
                params,
                self._truncate(json),
                self._truncate(data),
                len(content) if content else 0,
            )
        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=built_headers,
                params=params,
                json=json,
                data=data,
                content=content,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            if log_enabled:
                logger.error(
                    "HTTP request failed: method={} url={} params={} error={}",
                    method,
                    url,
                    params,
                    exc,
                )
            raise

        elapsed_ms = int((time.monotonic() - start) * 1000)
        body = response.text if response.content else ""
        if log_enabled:
            logger.info(
                "HTTP response: method={} url={} status={} elapsed_ms={}",
                method,
                url,
                response.status_code,
                elapsed_ms,
            )
            if body:
                snippet = body[:1000]
                if len(body) > 1000:
                    snippet = f"{snippet}...(truncated)"
                logger.info("HTTP response body: {}", snippet)

        if response.status_code >= 400 and log_enabled:
            logger.warning(
                "HTTP response error: method={} url={} status={} params={}",
                method,
                url,
                response.status_code,
                params,
            )
            if body:
                snippet = body[:500]
                if len(body) > 500:
                    snippet = f"{snippet}...(truncated)"
                logger.warning("HTTP response body: {}", snippet)

        if response_format == "raw":
            return response

        if response_format not in {"json", "yaml"}:
            raise ValueError(f"Unsupported response_format: {response_format}")

        from app.schemas.http_response import HttpResponse

        parsed_response = HttpResponse.from_response(response)
        # 保持现有语义：非成功业务码抛异常。
        parsed_data = parsed_response.data_or_raise()
        payload: Any = parsed_data
        if include_envelope:
            payload = {
                "code": parsed_response.code,
                "message": parsed_response.message,
                "data": parsed_data,
            }
            if parsed_response.timestamp is not None:
                payload["timestamp"] = parsed_response.timestamp

        if response_format == "json":
            return payload

        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)

    async def get(
            self,
            url: str,
            *,
            headers: Optional[Mapping[str, str]] = None,
            params: Optional[Mapping[str, Any]] = None,
            timeout: Optional[float] = None,
            response_format: Literal["raw", "json", "yaml"] = "raw",
            include_envelope: bool = False,
    ) -> Any:
        """发送 GET 请求。"""
        return await self.request(
            "GET",
            url,
            headers=headers,
            params=params,
            timeout=timeout,
            response_format=response_format,
            include_envelope=include_envelope,
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
            response_format: Literal["raw", "json", "yaml"] = "raw",
            include_envelope: bool = False,
    ) -> Any:
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
            response_format=response_format,
            include_envelope=include_envelope,
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
            response_format: Literal["raw", "json", "yaml"] = "raw",
            include_envelope: bool = False,
    ) -> Any:
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
            response_format=response_format,
            include_envelope=include_envelope,
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
            response_format: Literal["raw", "json", "yaml"] = "raw",
            include_envelope: bool = False,
    ) -> Any:
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
            response_format=response_format,
            include_envelope=include_envelope,
        )
