from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest

import app.utils.download_storage as download_storage
from app.core.exception.exceptions import ServiceException


def test_resolve_download_root_dir_raises_when_env_missing(monkeypatch):
    """
    测试目的：验证下载根目录未配置时会立即报错，避免回退到临时目录。
    预期结果：resolve_download_root_dir 抛出 ServiceException，且错误文案包含 FILE_DOWNLOAD_ROOT_DIR。
    """

    monkeypatch.delenv(download_storage.FILE_DOWNLOAD_ROOT_DIR_ENV, raising=False)

    with pytest.raises(ServiceException) as exc_info:
        download_storage.resolve_download_root_dir()

    assert download_storage.FILE_DOWNLOAD_ROOT_DIR_ENV in exc_info.value.message


def test_build_download_target_path_uses_date_tree_and_uuid_prefix(monkeypatch, tmp_path):
    """
    测试目的：验证目标路径遵循 yyyy/mm/dd 目录规则，并采用 uuid_原文件名 的命名格式。
    预期结果：返回路径位于指定日期目录下，文件名前缀为固定 UUID，且包含安全化后的原文件名。
    """

    monkeypatch.setenv(download_storage.FILE_DOWNLOAD_ROOT_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        download_storage.uuid,
        "uuid4",
        lambda: UUID("12345678-1234-5678-1234-567812345678"),
    )

    target = download_storage.build_download_target_path(
        "../../unsafe?.txt",
        now=datetime(2026, 3, 3, 10, 20, 30),
    )

    assert target.parent == tmp_path / "2026" / "03" / "03"
    assert target.name == "12345678-1234-5678-1234-567812345678_unsafe_.txt"


def test_safe_filename_sanitizes_path_and_illegal_chars():
    """
    测试目的：验证文件名清洗逻辑可消除路径穿越与非法字符风险。
    预期结果：输出仅保留安全文件名，非法字符被替换；无有效名称时回退默认文件名。
    """

    assert download_storage.safe_filename("../../a/b:c?.txt") == "b_c_.txt"
    assert download_storage.safe_filename("..") == download_storage.DEFAULT_SAFE_FILENAME
