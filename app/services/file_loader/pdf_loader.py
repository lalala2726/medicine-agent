from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader

from app.services.file_loader.base import FileLoader, ImageInfo, PageContent


def _filter_to_mime(filter_value: object) -> Optional[str]:
    if not filter_value:
        return None
    if isinstance(filter_value, list):
        filter_name = str(filter_value[0])
    else:
        filter_name = str(filter_value)

    mapping = {
        "/DCTDecode": "image/jpeg",
        "/JPXDecode": "image/jp2",
        "/FlateDecode": "image/png",
        "/CCITTFaxDecode": "image/tiff",
        "/JBIG2Decode": "image/jbig2",
    }
    return mapping.get(filter_name)


def _extract_page_images(page) -> List[ImageInfo]:
    # 遍历页面 XObject 资源，标注图片信息
    images: List[ImageInfo] = []
    resources = page.get("/Resources") if page else None
    if not resources:
        return images

    xobject = resources.get("/XObject") if resources else None
    if not xobject:
        return images

    for name in xobject:
        try:
            obj = xobject[name].get_object()
            if obj.get("/Subtype") != "/Image":
                continue
            width = obj.get("/Width")
            height = obj.get("/Height")
            mime_type = _filter_to_mime(obj.get("/Filter"))
            images.append(
                ImageInfo(
                    index=len(images) + 1,
                    name=str(name),
                    width=width,
                    height=height,
                    mime_type=mime_type,
                )
            )
        except Exception:
            continue
    return images


class PdfLoader(FileLoader):
    """PDF 解析器，按页提取文本和图片标注。"""

    def parse(self, file_path: Path) -> List[PageContent]:
        reader = PdfReader(str(file_path))
        pages: List[PageContent] = []
        for index, page in enumerate(reader.pages, start=1):
            # 每页分别提取文本，并标注图片
            text = page.extract_text() or ""
            images = _extract_page_images(page)
            pages.append(PageContent(page_number=index, text=text, images=images))
        return pages
