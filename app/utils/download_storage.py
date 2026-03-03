from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException

FILE_DOWNLOAD_ROOT_DIR_ENV = "FILE_DOWNLOAD_ROOT_DIR"
DEFAULT_SAFE_FILENAME = "downloaded_file"
_INVALID_FILENAME_PATTERN = re.compile(r'[<>:"|?*\x00-\x1f]')


def safe_filename(filename: str) -> str:
    """
    功能描述:
        对下载文件名进行安全化处理，防止路径穿越与非法字符导致落盘风险。

    参数说明:
        filename (str): 原始文件名，可能来自 URL 或响应头。

    返回值:
        str: 安全化后的文件名；当输入为空或非法时返回默认值 `downloaded_file`。

    异常说明:
        无。该函数不会主动抛出异常。
    """

    resolved = (filename or "").strip()
    basename = Path(resolved).name
    sanitized = basename.replace("/", "_").replace("\\", "_")
    sanitized = _INVALID_FILENAME_PATTERN.sub("_", sanitized)
    sanitized = sanitized.strip().strip(".")
    if not sanitized:
        return DEFAULT_SAFE_FILENAME
    return sanitized


def _ensure_writable_dir(path: Path) -> None:
    """
    功能描述:
        确保目标目录存在且可写，供下载文件落盘前执行目录可用性校验。

    参数说明:
        path (Path): 待校验目录路径。

    返回值:
        None: 校验成功时无返回值。

    异常说明:
        ServiceException:
            - 目录创建失败时抛出；
            - 目录不是有效目录或不可写时抛出。
    """

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"无法创建下载目录: {path}",
        ) from exc

    if not path.is_dir():
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"下载目录不是有效目录: {path}",
        )
    if not os.access(path, os.W_OK | os.X_OK):
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"下载目录不可写: {path}",
        )


def resolve_download_root_dir() -> Path:
    """
    功能描述:
        解析下载根目录配置并完成可写性校验。该配置为必填项。

    参数说明:
        无。

    返回值:
        Path: 生效的下载根目录绝对路径。

    异常说明:
        ServiceException:
            - 未配置 `FILE_DOWNLOAD_ROOT_DIR` 时抛出；
            - 目录不可创建或不可写时抛出。
    """

    raw_value = (os.getenv(FILE_DOWNLOAD_ROOT_DIR_ENV) or "").strip()
    if not raw_value:
        raise ServiceException(
            code=ResponseCode.INTERNAL_ERROR,
            message=f"{FILE_DOWNLOAD_ROOT_DIR_ENV} is not set",
        )

    root_dir = Path(raw_value).expanduser()
    _ensure_writable_dir(root_dir)
    return root_dir


def build_download_target_path(
    filename: str,
    now: datetime | None = None,
) -> Path:
    """
    功能描述:
        基于固定下载根目录构造文件落盘路径，目录按 `yyyy/mm/dd` 分层，
        文件名按 `uuid_原文件名` 生成。

    参数说明:
        filename (str): 原始文件名。
        now (datetime | None): 时间戳注入点，默认值为 None；为空时使用当前本地时间。

    返回值:
        Path: 最终下载落盘路径（文件可能尚未写入）。

    异常说明:
        ServiceException:
            - 下载根目录未配置时抛出；
            - 日期目录创建失败或不可写时抛出。
    """

    root_dir = resolve_download_root_dir()
    resolved_now = now or datetime.now()
    date_dir = root_dir / f"{resolved_now.year:04d}" / f"{resolved_now.month:02d}" / f"{resolved_now.day:02d}"
    _ensure_writable_dir(date_dir)

    resolved_filename = safe_filename(filename)
    unique_prefix = str(uuid.uuid4())
    return date_dir / f"{unique_prefix}_{resolved_filename}"


__all__ = [
    "FILE_DOWNLOAD_ROOT_DIR_ENV",
    "build_download_target_path",
    "resolve_download_root_dir",
    "safe_filename",
]
