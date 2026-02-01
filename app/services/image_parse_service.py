import json
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.llm import create_chat_model
from app.core.prompts import DRUG_PARSER_PROMPT
from app.utils.file_utils import FileUtils


def parse_drug_images(images: List[str]) -> dict:
    def normalize_image_data(image_str: str) -> str:
        trimmed = image_str.strip()
        lower = trimmed.lower()
        if lower.startswith("data:image"):
            return trimmed
        if lower.startswith("http://") or lower.startswith("https://"):
            return FileUtils.image_url_to_base64_data_url(trimmed)
        return f"data:image/png;base64,{trimmed}"

    image_parts = [
        {"type": "image_url", "image_url": {"url": normalize_image_data(img)}}
        for img in images
    ]

    model = create_chat_model(
        model="qwen3-vl-plus",
        response_format={"type": "json_object"},
    )
    messages = [
        SystemMessage(content=DRUG_PARSER_PROMPT),
        HumanMessage(content=image_parts),
    ]
    result = model.invoke(messages)

    try:
        return json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise ServiceException(
            message="模型返回非 JSON 内容",
            code=ResponseCode.INTERNAL_ERROR,
        ) from exc
