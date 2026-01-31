import json
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.llm import create_chat_model
from app.core.prompts import DRUG_PARSER_PROMPT


def parse_drug_images(images: List[str]) -> dict:
    def normalize_image_data(image_str: str) -> str:
        if image_str.startswith("data:image"):
            return image_str
        return f"data:image/png;base64,{image_str}"

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
