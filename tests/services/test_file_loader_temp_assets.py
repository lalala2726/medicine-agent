from app.core.file_loader import base


def test_create_temp_image_dir_uses_env(monkeypatch, tmp_path):
    # 指定图片父目录时，临时目录应创建在该目录下
    monkeypatch.setenv("FILE_IMAGE_OUTPUT_DIR", str(tmp_path))
    image_dir = base.create_temp_image_dir("demo")
    assert image_dir.exists()
    assert image_dir.parent == tmp_path


def test_register_and_cleanup_temp_assets(tmp_path):
    # 注册临时资源后，按文件名清理应同时删除图片目录与源文件
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    source_path = tmp_path / "source.txt"
    source_path.write_text("data", encoding="utf-8")

    base.register_temp_assets("source.txt", image_dir, source_path)
    result = base.cleanup_temp_assets("source.txt")

    assert result["deleted_count"] == 1
    assert not image_dir.exists()
    assert not source_path.exists()
