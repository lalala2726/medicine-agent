import pytest

from app.core.exception.exceptions import ServiceException
from app.rag.file_loader.detectors.url_extension import validate_url_extension


def test_validate_url_extension_accepts_supported_suffix() -> None:
    """
    测试目的：验证 URL 后缀在支持列表中时可通过第一阶段校验。
    预期结果：返回标准化后缀字符串。
    """
    suffix = validate_url_extension("https://example.com/docs/manual.PDF")
    assert suffix == ".pdf"


def test_validate_url_extension_rejects_missing_suffix() -> None:
    """
    测试目的：验证 URL 不含文件后缀时会被第一阶段校验拒绝。
    预期结果：抛出 ServiceException。
    """
    with pytest.raises(ServiceException):
        validate_url_extension("https://example.com/download")


def test_validate_url_extension_rejects_unsupported_suffix() -> None:
    """
    测试目的：验证 URL 后缀不在支持列表中时会被第一阶段校验拒绝。
    预期结果：抛出 ServiceException。
    """
    with pytest.raises(ServiceException):
        validate_url_extension("https://example.com/archive.bin")
