from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.client.domain.tools.card_tools import build_card_response
from app.core.agent.agent_event_bus import enqueue_final_sse_response
from app.core.config_sync import AgentChatModelSlot, create_agent_chat_llm
from app.schemas.sse_response import Card
from app.utils.list_utils import TextListUtils

# 诊断问卷卡片类型标识。
CONSULTATION_QUESTIONNAIRE_CARD_TYPE = "consultation-questionnaire-card"
# 诊断问卷卡工具固定成功返回值。
QUESTIONNAIRE_CARD_TOOL_SUCCESS_RESULT = "__SUCCESS__"
# 诊断问卷卡问题数量上限。
MAX_CONSULTATION_QUESTIONNAIRE_COUNT = 5
# 诊断问卷选项生成阶段固定使用的模型名称。
CONSULTATION_QUESTIONNAIRE_OPTION_MODEL_NAME = "qwen-flash"
# 诊断问卷选项生成系统提示词模板。
CONSULTATION_QUESTIONNAIRE_OPTION_SYSTEM_PROMPT_TEMPLATE = (
    SystemMessagePromptTemplate.from_template(
        """
            你是医学诊断问卷选项生成助手。

            你的任务是根据输入的问题文本，直接生成完整的诊断问卷 JSON。

            输出规则：
            1. 只输出一个 JSON 对象，禁止输出解释、标题、Markdown 或其他文本。
            2. 顶层格式固定为 {{"questions": [{{"question": "问题文本", "options": ["选项1", "选项2"]}}]}}。
            3. `questions` 数组长度必须和输入问题数量完全一致，按原顺序一一对应。
            4. `question` 字段必须原样使用输入中的问题文本，不要改写。
            5. 每个问题输出 2 到 5 个选项。
            6. `options` 中的选项必须是简短中文短语，适合按钮点击，不要输出整句解释。
            7. 选项之间必须可区分，且要尽量覆盖常见回答，例如“有”“没有”“不确定”“偶尔”“持续”等。
            8. 只能输出一个 JSON 对象，结尾不要补充任何额外字符。

            输入示例：
            ["这两天有发热吗？", "有没有咳嗽？"]

            正确输出示例：
            {{
              "questions": [
                {{
                  "question": "这两天有发热吗？",
                  "options": ["有", "没有", "不确定"]
                }},
                {{
                  "question": "有没有咳嗽？",
                  "options": ["有", "没有", "偶尔"]
                }}
              ]
            }}

            待处理问题 JSON 数组：
            {questions_json}
        """.strip()
    )
)


class ConsultationQuestionnaireQuestionItem(BaseModel):
    """诊断问卷卡中的单个问题项。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, description="问题文本。")
    options: list[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="该问题对应的可点击选项列表。",
    )

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        """校验单个问题文本。

        Args:
            value: 原始问题文本。

        Returns:
            str: 去空白后的问题文本。
        """

        normalized_questions = TextListUtils.normalize_unique_required(
            [value],
            field_name="question",
        )
        return normalized_questions[0]

    @field_validator("options")
    @classmethod
    def _validate_options(cls, value: list[str]) -> list[str]:
        """校验单题选项列表。

        Args:
            value: 原始选项列表。

        Returns:
            list[str]: 去空白且不重复的有效选项列表。
        """

        return TextListUtils.normalize_unique_required(
            value,
            field_name="options",
        )


class ConsultationQuestionnaireCardData(BaseModel):
    """诊断问卷卡片数据。"""

    model_config = ConfigDict(extra="forbid")

    questions: list[ConsultationQuestionnaireQuestionItem] = Field(
        ...,
        min_length=1,
        max_length=MAX_CONSULTATION_QUESTIONNAIRE_COUNT,
        description="诊断问卷问题列表。",
    )


class SendConsultationQuestionnaireCardRequest(BaseModel):
    """发送诊断问卷卡片工具参数。"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "questions": [
                    "这两天有发热吗？",
                    "有没有咳嗽？",
                    "吞咽时疼痛明显吗？",
                ]
            }
        },
    )

    questions: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_CONSULTATION_QUESTIONNAIRE_COUNT,
        description="需要发送给前端问卷卡的问题文本列表，保持原顺序传入。",
    )

    @field_validator("questions")
    @classmethod
    def _validate_questions(cls, value: list[str]) -> list[str]:
        """校验问题文本列表。

        Args:
            value: 原始问题文本列表。

        Returns:
            list[str]: 去空白且去重后的问题文本列表。
        """

        return TextListUtils.normalize_required(
            value,
            field_name="questions",
        )


def _parse_questionnaire_card_data(
        raw_value: Any,
        *,
        question_count: int,
) -> ConsultationQuestionnaireCardData:
    """解析问卷卡生成模型输出。

    Args:
        raw_value: 模型返回的原始 JSON 结构。
        question_count: 当前问题总数。

    Returns:
        ConsultationQuestionnaireCardData: 校验通过的诊断问卷卡片数据。

    Raises:
        ValueError: 返回结构非法或数量不匹配时抛出。
    """

    card_data = ConsultationQuestionnaireCardData.model_validate(raw_value)
    if len(card_data.questions) != question_count:
        raise ValueError("diagnosis questionnaire questions 数量必须与输入问题数量一致")
    return card_data


def _strip_json_code_fence(raw_text: str) -> str:
    """移除模型输出外层 Markdown JSON 代码块。

    Args:
        raw_text: 模型原始输出文本。

    Returns:
        str: 去除外层代码块后的文本。
    """

    normalized_text = raw_text.strip()
    if not normalized_text.startswith("```"):
        return normalized_text

    lines = normalized_text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip()
    return normalized_text


def _load_first_json_value(raw_text: str) -> Any:
    """从模型输出中解析第一个 JSON 值。

    Args:
        raw_text: 模型原始输出文本。

    Returns:
        Any: 解析得到的 JSON 值。

    Raises:
        ValueError: 文本为空时抛出。
        json.JSONDecodeError: JSON 结构非法时抛出。
    """

    normalized_text = _strip_json_code_fence(raw_text)
    if not normalized_text:
        raise ValueError("diagnosis questionnaire options 返回内容不能为空")

    try:
        return json.loads(normalized_text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed_value, _ = decoder.raw_decode(normalized_text)
        return parsed_value


def _generate_questionnaire_card_data(
        questions: list[str],
) -> ConsultationQuestionnaireCardData:
    """使用额外 LLM 生成完整诊断问卷卡片数据。

    Args:
        questions: 已归一化的问题文本列表。

    Returns:
        ConsultationQuestionnaireCardData: 可直接发送给前端的问卷卡片数据。
    """

    llm = create_agent_chat_llm(
        slot=AgentChatModelSlot.CLIENT_CONSULTATION_FINAL_DIAGNOSIS,
        model_name=CONSULTATION_QUESTIONNAIRE_OPTION_MODEL_NAME,
        temperature=0.0,
        think=False,
    )
    system_message = CONSULTATION_QUESTIONNAIRE_OPTION_SYSTEM_PROMPT_TEMPLATE.format(
        questions_json=json.dumps(questions, ensure_ascii=False),
    )
    response = llm.invoke([system_message])
    response_text = str(getattr(response, "content", "") or "").strip()
    raw_value = _load_first_json_value(response_text)
    return _parse_questionnaire_card_data(
        raw_value,
        question_count=len(questions),
    )


@tool(
    args_schema=SendConsultationQuestionnaireCardRequest,
    description=(
            "向前端发送诊断问卷卡。"
            "调用时机：当你已经整理出 2 到 5 个需要连续追问的自然语言问题，"
            "希望用户在前端按问卷方式逐题作答并最终一次性提交时。"
            "调用前必须先把问题整理成自然语言问题文本列表，不要传症状名。"
    ),
)
def send_consultation_questionnaire_card(questions: list[str]) -> str:
    """构建并发送诊断问卷卡。

    Args:
        questions: 需要发送给前端的自然语言问题文本列表。

    Returns:
        str: 固定成功标记字符串。
    """

    normalized_questions = TextListUtils.normalize_required(
        questions,
        field_name="questions",
    )
    if len(normalized_questions) > MAX_CONSULTATION_QUESTIONNAIRE_COUNT:
        raise ValueError(
            f"questions 最多只能包含 {MAX_CONSULTATION_QUESTIONNAIRE_COUNT} 个问题"
        )

    card_data = _generate_questionnaire_card_data(normalized_questions)
    enqueue_final_sse_response(
        build_card_response(
            Card(
                type=CONSULTATION_QUESTIONNAIRE_CARD_TYPE,
                data=card_data.model_dump(mode="json", exclude_none=True),
            ),
            persist_card=True,
        )
    )
    return QUESTIONNAIRE_CARD_TOOL_SUCCESS_RESULT


__all__ = [
    "CONSULTATION_QUESTIONNAIRE_CARD_TYPE",
    "CONSULTATION_QUESTIONNAIRE_OPTION_MODEL_NAME",
    "ConsultationQuestionnaireCardData",
    "ConsultationQuestionnaireQuestionItem",
    "MAX_CONSULTATION_QUESTIONNAIRE_COUNT",
    "QUESTIONNAIRE_CARD_TOOL_SUCCESS_RESULT",
    "SendConsultationQuestionnaireCardRequest",
    "send_consultation_questionnaire_card",
]
