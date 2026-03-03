from __future__ import annotations

import pytest

import app.utils.file_utils as file_utils_module
from app.core.exception.exceptions import ServiceException
from app.utils.download_storage import FILE_DOWNLOAD_ROOT_DIR_ENV
from app.utils.file_utils import FileUtils


class _FakeResponse:
    def __init__(self, payload: bytes, headers: dict[str, str]) -> None:
        self._payload = payload
        self._offset = 0
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_download_file_writes_to_fixed_target_path(monkeypatch, tmp_path):
    """
    测试目的：验证 download_file 使用固定目录目标路径写入文件且返回正确文件名。
    预期结果：文件写入成功，返回路径与目标路径一致，文件内容与响应体一致。
    """

    target_path = tmp_path / "2026" / "03" / "03" / "fixed_download.txt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        file_utils_module,
        "build_download_target_path",
        lambda _filename: target_path,
    )
    monkeypatch.setattr(
        file_utils_module,
        "urlopen",
        lambda _url, timeout: _FakeResponse(
            b"hello world",
            {
                "Content-Type": "text/plain",
                "Content-Disposition": "attachment; filename*=UTF-8''doc.txt",
            },
        ),
    )

    filename, saved_path = FileUtils.download_file("http://example.com/download")

    assert filename == "doc.txt"
    assert saved_path == target_path
    assert saved_path.read_bytes() == b"hello world"


def test_download_file_raises_when_download_root_not_configured(monkeypatch):
    """
    测试目的：验证未配置 FILE_DOWNLOAD_ROOT_DIR 时下载流程会直接抛出配置异常。
    预期结果：download_file 抛出 ServiceException，且错误文案包含 FILE_DOWNLOAD_ROOT_DIR。
    """

    monkeypatch.delenv(FILE_DOWNLOAD_ROOT_DIR_ENV, raising=False)
    monkeypatch.setattr(
        file_utils_module,
        "urlopen",
        lambda _url, timeout: _FakeResponse(
            b"payload",
            {
                "Content-Type": "text/plain",
                "Content-Disposition": "attachment; filename=test.txt",
            },
        ),
    )

    with pytest.raises(ServiceException) as exc_info:
        FileUtils.download_file("http://example.com/download")

    assert FILE_DOWNLOAD_ROOT_DIR_ENV in exc_info.value.message
