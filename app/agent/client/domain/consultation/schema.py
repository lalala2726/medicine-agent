from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.client.domain.consultation.state import (
    CONSULTATION_STATUS_COLLECTING,
    CONSULTATION_STATUS_COMPLETED,
    ConsultationStatusValue,
)


class ConsultationStatusSchema(BaseModel):
    """consultation 状态判断结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    should_enter_diagnosis: bool = Field(description="当前是否已经具备进入最终诊断节点的条件。")
    consultation_status: ConsultationStatusValue = Field(
        description="当前病情咨询阶段，只允许 collecting 或 completed。",
    )


class ConsultationQuestionSchema(BaseModel):
    """consultation 问询卡片结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    should_enter_diagnosis: bool = Field(description="当前是否已经具备进入最终诊断节点的条件。")
    consultation_status: ConsultationStatusValue = Field(
        description="当前病情咨询阶段，只允许 collecting 或 completed。",
    )
    question_text: str = Field(description="给用户展示的追问文本。")
    options: list[str] = Field(default_factory=list, description="选择卡片选项列表。")

    @field_validator("question_text")
    @classmethod
    def _normalize_question_text(cls, value: str) -> str:
        """清理追问文本。"""

        normalized = value.strip()
        if not normalized:
            raise ValueError("question_text 不能为空")
        return normalized

    @field_validator("options")
    @classmethod
    def _normalize_options(cls, value: list[str]) -> list[str]:
        """清理并去重选项列表。"""

        normalized_options: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = str(item or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_options.append(normalized)
        return normalized_options

    @model_validator(mode="after")
    def _validate_collecting_payload(self) -> "ConsultationQuestionSchema":
        """校验 collecting / completed 两类输出约束。"""

        if self.should_enter_diagnosis:
            self.consultation_status = CONSULTATION_STATUS_COMPLETED
            self.options = []
            return self
        self.consultation_status = CONSULTATION_STATUS_COLLECTING
        if len(self.options) < 2:
            raise ValueError("collecting 状态至少需要 2 个选项")
        if len(self.options) > 4:
            self.options = self.options[:4]
        return self


class ConsultationFinalDiagnosisSchema(BaseModel):
    """consultation 最终诊断结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    diagnosis_text: str = Field(description="最终给用户展示的诊断与就医建议文本。")
    should_recommend_products: bool = Field(description="当前是否需要继续搜索并推荐商城药品。")
    product_keyword: str | None = Field(default=None, description="用于商品搜索的关键词，可为空。")
    product_usage: str | None = Field(default=None, description="用于商品搜索的适用场景，可为空。")

    @field_validator("diagnosis_text")
    @classmethod
    def _normalize_diagnosis_text(cls, value: str) -> str:
        """清理诊断文本。"""

        normalized = value.strip()
        if not normalized:
            raise ValueError("diagnosis_text 不能为空")
        return normalized

    @field_validator("product_keyword", "product_usage")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        """清理可选搜索参数。"""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


__all__ = [
    "ConsultationFinalDiagnosisSchema",
    "ConsultationQuestionSchema",
    "ConsultationStatusSchema",
]
