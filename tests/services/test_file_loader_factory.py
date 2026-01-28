from pathlib import Path

from app.services.file_loader.base import cleanup_temp_assets
from app.services.file_loader.factory import FileLoaderFactory


def test_parse_file_with_images_creates_dir_and_registers(monkeypatch, tmp_path):
    # 解析文本文件时应创建临时图片目录，并可按文件名清理
    monkeypatch.setenv("FILE_IMAGE_OUTPUT_DIR", str(tmp_path))
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    parsed = FileLoaderFactory.parse_file_with_images(file_path, source_name="sample.txt")
    image_dir = Path(parsed["image_dir"])

    assert image_dir.exists()
    assert parsed["pages"][0]["text"].strip() == "hello"

    cleanup = cleanup_temp_assets("sample.txt")
    assert cleanup["deleted_count"] == 1
    assert not image_dir.exists()
    assert not file_path.exists()
