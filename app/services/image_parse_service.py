import json
from typing import Any, List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from app.core.config_sync import create_agent_image_llm
from app.core.codes import ResponseCode
from app.core.exception.exceptions import ServiceException
from app.utils.file_utils import FileUtils
from app.utils.prompt_utils import load_prompt

_DRUG_PARSER_PROMPT = load_prompt("image_parser/drug_prompt.md")


class DrugImageSchema(BaseModel):
    """
    功能描述：
        药品图片结构化识别结果 Schema，约束图片解析返回字段与类型。

    参数说明：
        commonName (str | None): 药品通用名。
        brand (str | None): 品牌名称。
        composition (str | None): 成分信息。
        characteristics (str | None): 性状描述。
        packaging (str | None): 包装规格。
        validityPeriod (str | None): 有效期。
        storageConditions (str | None): 贮藏条件。
        productionUnit (str | None): 生产单位。
        approvalNumber (str | None): 批准文号。
        executiveStandard (str | None): 执行标准。
        originType (str | None): 产地类型（国产/进口）。
        isOutpatientMedicine (bool | None): 是否外用药。
        prescription (bool | None): 是否处方药。
        efficacy (str | None): 功能主治。
        usageMethod (str | None): 用法用量。
        adverseReactions (str | None): 不良反应。
        precautions (str | None): 注意事项。
        taboo (str | None): 禁忌。
        warmTips (str | None): 温馨提示。
        instruction (str | None): 说明书文本。

    返回值：
        无（数据模型定义）。

    异常说明：
        pydantic.ValidationError: 当字段类型不符合约束时抛出。
    """

    commonName: str | None = Field(default=None)
    brand: str | None = Field(default=None)
    composition: str | None = Field(default=None)
    characteristics: str | None = Field(default=None)
    packaging: str | None = Field(default=None)
    validityPeriod: str | None = Field(default=None)
    storageConditions: str | None = Field(default=None)
    productionUnit: str | None = Field(default=None)
    approvalNumber: str | None = Field(default=None)
    executiveStandard: str | None = Field(default=None)
    originType: str | None = Field(default=None)
    isOutpatientMedicine: bool | None = Field(default=None)
    prescription: bool | None = Field(default=None)
    efficacy: str | None = Field(default=None)
    usageMethod: str | None = Field(default=None)
    adverseReactions: str | None = Field(default=None)
    precautions: str | None = Field(default=None)
    taboo: str | None = Field(default=None)
    warmTips: str | None = Field(default=None)
    instruction: str | None = Field(default=None)


def parse_drug_images(images: List[str]) -> dict:
    """
    功能描述：
        解析药品图片并返回结构化字段结果。

    参数说明：
        images (List[str]): 图片列表，支持 URL、DataURL 或纯 Base64 字符串。

    返回值：
        dict: 药品图片结构化识别结果字典，字段由 `DrugImageSchema` 定义。

    异常说明：
        ServiceException: 当模型结构化输出不合法时抛出 `INTERNAL_ERROR`。
    """

    def normalize_image_data(image_str: str) -> str:
        """
        功能描述：
            规范化单张图片输入，统一转换为可直接供多模态模型消费的 Data URL。

        参数说明：
            image_str (str): 原始图片输入字符串。

        返回值：
            str: 规范化后的 Data URL 字符串。

        异常说明：
            无。URL 下载或转换异常由 `FileUtils.image_url_to_base64_data_url` 向上抛出。
        """

        trimmed = image_str.strip()
        lower = trimmed.lower()
        if lower.startswith("data:image"):
            return trimmed
        if lower.startswith("http://") or lower.startswith("https://"):
            return FileUtils.image_url_to_base64_data_url(trimmed)
        return f"data:image/png;base64,{trimmed}"

    def extract_text_from_content(content: Any) -> str:
        """
        功能描述：
            将不同 SDK 返回的文本内容统一提取为字符串。

        参数说明：
            content (Any): SDK 返回的消息内容。

        返回值：
            str: 提取出的文本内容。

        异常说明：
            ValueError: 当内容为空或不包含可用文本时抛出。
        """

        if isinstance(content, str):
            normalized = content.strip()
            if normalized:
                return normalized
            raise ValueError("content is empty")

        if not isinstance(content, list):
            raise ValueError("content is not a supported type")

        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    text_parts.append(normalized)
                continue

            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)

            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())

        if not text_parts:
            raise ValueError("content does not contain text")
        return "".join(text_parts)

    def extract_chat_completion_text(completion: Any) -> str:
        """
        功能描述：
            从 OpenAI 兼容 chat completion 结果中提取文本。

        参数说明：
            completion (Any): `chat.completions.create` 的响应对象。

        返回值：
            str: 模型输出文本。

        异常说明：
            ValueError: 当响应缺少可用文本时抛出。
        """

        choices = getattr(completion, "choices", None) or []
        if not choices:
            if hasattr(completion, "content"):
                return extract_text_from_content(getattr(completion, "content", None))
            raise ValueError("completion choices are empty")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("completion message is missing")
        return extract_text_from_content(getattr(message, "content", None))

    normalized_images = [normalize_image_data(img) for img in images]

    try:
        llm = create_agent_image_llm(
            think=False,
        )
        completion = llm.invoke(
            [
                HumanMessage(
                    content=[
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": image},
                            }
                            for image in normalized_images
                        ],
                        {
                            "type": "text",
                            "text": _DRUG_PARSER_PROMPT,
                        },
                    ],
                )
            ],
        )
        content = extract_chat_completion_text(completion)

        return DrugImageSchema.model_validate(json.loads(content)).model_dump()
    except (ValidationError, ValueError, TypeError) as exc:
        raise ServiceException(
            message="模型返回非 JSON 内容",
            code=ResponseCode.INTERNAL_ERROR,
        ) from exc
