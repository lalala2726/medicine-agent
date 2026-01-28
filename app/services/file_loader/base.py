from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Optional


@dataclass
class ImageInfo:
    """
    图片基础信息，用于标注页面中的图片。

    Attributes:
        index: 图片在文档中的顺序索引
        name: 图片文件名
        width: 图片宽度（像素）
        height: 图片高度（像素）
        mime_type: 图片 MIME 类型
        path: 图片保存路径
    """

    index: int
    name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: Optional[str] = None
    path: Optional[str] = None


@dataclass
class PageContent:
    """
    页面内容结构，统一输出格式。

    Attributes:
        page_number: 页码（从 1 开始）
        text: 页面文本内容
        images: 页面中的图片列表
        page_label: 页面标签（如 Excel 的 sheet 名称）
    """

    page_number: int
    text: str
    images: List[ImageInfo] = field(default_factory=list)
    page_label: Optional[str] = None

    def to_dict(self) -> dict:
        """
        转换为可序列化的字典结构，方便打印或存储。

        Returns:
            包含页面信息和图片统计的字典
        """
        return {
            "page_number": self.page_number,
            "page_label": self.page_label,
            "text": self.text,
            "images": [asdict(image) for image in self.images],
            "has_images": bool(self.images),
            "image_count": len(self.images),
        }


class FileLoader(ABC):
    """文件解析器抽象基类，所有解析器需实现 parse。"""

    @abstractmethod
    def parse(
        self, file_path: Path, output_dir: Optional[Path] = None
    ) -> List[PageContent]:
        """
        解析文件并返回按页分组的内容。

        Args:
            file_path: 文件路径
            output_dir: 图片输出目录（可选）

        Returns:
            按页组织的内容列表
        """
        raise NotImplementedError


# 默认图片输出目录，可通过环境变量 FILE_IMAGE_OUTPUT_DIR 覆盖
DEFAULT_IMAGE_OUTPUT_DIR = Path(
    os.getenv("FILE_IMAGE_OUTPUT_DIR", "/Users/zhangchuang/Downloads/testImages")
)


def ensure_image_output_dir(output_dir: Optional[Path] = None) -> Path:
    """
    确保图片输出目录存在，统一在此创建。

    Args:
        output_dir: 指定的输出目录（可选）

    Returns:
        确保存在的输出目录路径
    """
    target_dir = output_dir or DEFAULT_IMAGE_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def create_temp_image_dir(file_stem: Optional[str] = None) -> Path:
    """
    为单个文件创建临时图片目录，便于后续处理与清理。

    Args:
        file_stem: 文件名（不含扩展名），用作目录名前缀

    Returns:
        创建的临时目录路径
    """
    base_dir = os.getenv("FILE_IMAGE_OUTPUT_DIR")
    prefix = f"{file_stem}_" if file_stem else "import_images_"
    if base_dir:
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        # 在指定目录下创建临时子目录
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_path)))
    # 使用系统默认临时目录
    return Path(tempfile.mkdtemp(prefix=prefix))


@dataclass
class TempAssetInfo:
    """
    临时资源信息，用于清理解析产生的文件与图片目录。

    Attributes:
        filename: 文件名
        image_dir: 图片目录路径
        source_path: 源文件路径（可选）
    """

    filename: str
    image_dir: str
    source_path: Optional[str] = None


_TEMP_ASSET_REGISTRY: Dict[str, List[TempAssetInfo]] = {}  # 全局临时资源注册表


def register_temp_assets(
    filename: str, image_dir: Path, source_path: Optional[Path] = None
) -> None:
    """
    注册解析产生的临时资源，便于后续按文件名清理。

    Args:
        filename: 文件名（用作清理时的键）
        image_dir: 图片目录路径
        source_path: 源文件路径（可选）
    """
    record = TempAssetInfo(
        filename=filename,
        image_dir=str(image_dir),
        source_path=str(source_path) if source_path else None,
    )
    # 将记录添加到对应文件名的列表中
    _TEMP_ASSET_REGISTRY.setdefault(filename, []).append(record)


def cleanup_temp_assets(filename: str) -> dict:
    """
    按文件名清理临时资源（原文件与图片目录）。

    Args:
        filename: 文件名

    Returns:
        清理结果统计字典
    """
    items = _TEMP_ASSET_REGISTRY.pop(filename, [])
    removed_images: List[str] = []
    removed_files: List[str] = []

    for item in items:
        # 清理下载的原始文件
        if item.source_path:
            source_path = Path(item.source_path)
            if source_path.exists():
                try:
                    source_path.unlink()
                    removed_files.append(str(source_path))
                except Exception:
                    pass
        # 清理解析生成的图片目录
        if item.image_dir:
            image_dir = Path(item.image_dir)
            if image_dir.exists():
                try:
                    shutil.rmtree(image_dir, ignore_errors=True)
                    removed_images.append(str(image_dir))
                except Exception:
                    pass

    return {
        "filename": filename,
        "deleted_files": removed_files,
        "deleted_image_dirs": removed_images,
        "deleted_count": len(items),
    }


def _normalize_extension(extension: Optional[str]) -> str:
    """
    统一扩展名格式，保证以 '.' 开头。

    Args:
        extension: 原始扩展名

    Returns:
        标准化后的扩展名（以点号开头）
    """
    if not extension:
        return ".img"  # 默认扩展名
    if extension.startswith("."):
        return extension
    return f".{extension}"


def save_image_bytes(
    output_dir: Path,
    file_stem: str,
    page_number: int,
    index: int,
    data: bytes,
    extension: Optional[str],
) -> Path:
    """
    将图片二进制内容保存到指定目录并返回路径。

    Args:
        output_dir: 输出目录
        file_stem: 文件名（不含扩展名）
        page_number: 页码
        index: 图片索引
        data: 图片二进制数据
        extension: 图片扩展名

    Returns:
        保存后的图片路径
    """
    target_dir = ensure_image_output_dir(output_dir)
    safe_extension = _normalize_extension(extension)
    # 生成文件名格式：{文件名}_p{页码}_img{索引}.{扩展名}
    filename = f"{file_stem}_p{page_number}_img{index}{safe_extension}"
    target_path = target_dir / filename
    target_path.write_bytes(data)
    return target_path


def copy_image_file(
    output_dir: Path,
    source_path: Path,
    file_stem: str,
    page_number: int,
    index: int,
) -> Path:
    """
    复制原始图片文件到输出目录，便于统一管理。

    Args:
        output_dir: 输出目录
        source_path: 源图片路径
        file_stem: 文件名（不含扩展名）
        page_number: 页码
        index: 图片索引

    Returns:
        复制后的图片路径
    """
    target_dir = ensure_image_output_dir(output_dir)
    extension = _normalize_extension(source_path.suffix)
    # 生成文件名格式：{文件名}_p{页码}_img{索引}.{扩展名}
    filename = f"{file_stem}_p{page_number}_img{index}{extension}"
    target_path = target_dir / filename
    shutil.copy2(source_path, target_path)
    return target_path
