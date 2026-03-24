from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.client.domain.consultation.state import (
    ConsultationModeValue,
    ConsultationNextActionValue,
)
from app.utils.list_utils import TextListUtils


class ConsultationRouteSchema(BaseModel):
    """consultation 子图内路由节点结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    next_action: ConsultationNextActionValue = Field(description="本轮 consultation 的下一步动作。")
    consultation_mode: ConsultationModeValue = Field(description="当前会话属于简单医学询问还是诊断型咨询。")
    reason: str = Field(description="路由节点给出的简要原因说明。")

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, value: str) -> str:
        """
        功能描述：
            清理路由原因文本的首尾空白。

        参数说明：
            value (str): 原始原因文本。

        返回值：
            str: 清理后的原因文本。

        异常说明：
            无。
        """

        normalized = str(value or "").strip()
        return normalized or "未说明原因"


class ConsultationQuestionSchema(BaseModel):
    """consultation 追问节点结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    diagnosis_ready: bool = Field(description="当前信息是否已经足够进入最终诊断。")
    question_reply_text: str | None = Field(default=None, description="继续追问前展示给用户的阶段性分析文本。")
    question_text: str | None = Field(default=None, description="选择卡片标题使用的核心问题。")
    options: list[str] = Field(default_factory=list, description="用于选择卡片的互斥选项列表。")
    slot_key: str | None = Field(default=None, description="当前追问对应的结构化槽位标识，用于去重追问。")

    @field_validator("question_reply_text", "question_text", "slot_key")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        """
        功能描述：
            清理追问节点可选文本字段的首尾空白。

        参数说明：
            value (str | None): 原始文本值。

        返回值：
            str | None: 归一化后的文本；空串统一返回 `None`。

        异常说明：
            无。
        """

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("options")
    @classmethod
    def _normalize_options(cls, value: list[str]) -> list[str]:
        """
        功能描述：
            清理、去重并保留选项原顺序。

        参数说明：
            value (list[str]): 原始选项列表。

        返回值：
            list[str]: 归一化后的选项列表。

        异常说明：
            无。
        """

        return TextListUtils.normalize(value)

    @model_validator(mode="after")
    def _validate_shape(self) -> "ConsultationQuestionSchema":
        """
        功能描述：
            校验“继续追问”和“信息已足够”两种输出形态。

        参数说明：
            无。

        返回值：
            ConsultationQuestionSchema: 校验后的模型自身。

        异常说明：
            ValueError: 输出字段不符合当前分支约束时抛出。
        """

        if self.diagnosis_ready:
            self.question_reply_text = None
            self.question_text = None
            self.options = []
            self.slot_key = None
            return self

        if not self.question_reply_text:
            raise ValueError("diagnosis_ready=false 时 question_reply_text 不能为空")
        if not self.question_text:
            raise ValueError("diagnosis_ready=false 时 question_text 不能为空")
        if not self.slot_key:
            raise ValueError("diagnosis_ready=false 时 slot_key 不能为空")
        if len(self.options) < 2:
            raise ValueError("diagnosis_ready=false 时 options 至少需要 2 项")
        if len(self.options) > 4:
            self.options = self.options[:4]
        return self


__all__ = [
    "ConsultationRouteSchema",
    "ConsultationQuestionSchema",
]
