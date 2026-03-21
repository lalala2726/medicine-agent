from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConsultationQuestionSchema(BaseModel):
    """consultation 追问节点结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    diagnosis_ready: bool = Field(description="当前信息是否已经足够进入最终诊断。")
    question_reply_text: str | None = Field(default=None, description="继续追问前展示给用户的阶段性分析文本。")
    question_text: str | None = Field(default=None, description="选择卡片标题使用的核心问题。")
    options: list[str] = Field(default_factory=list, description="用于选择卡片的互斥选项列表。")

    @field_validator("question_reply_text", "question_text")
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

        normalized_options: list[str] = []
        seen: set[str] = set()
        for raw_option in value:
            normalized = str(raw_option or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_options.append(normalized)
        return normalized_options

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
            return self

        if not self.question_reply_text:
            raise ValueError("diagnosis_ready=false 时 question_reply_text 不能为空")
        if not self.question_text:
            raise ValueError("diagnosis_ready=false 时 question_text 不能为空")
        if len(self.options) < 2:
            raise ValueError("diagnosis_ready=false 时 options 至少需要 2 项")
        if len(self.options) > 4:
            self.options = self.options[:4]
        return self


class ConsultationFinalDiagnosisSchema(BaseModel):
    """consultation 最终诊断节点结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    diagnosis_text: str = Field(description="最终诊断阶段返回给用户的完整建议文本。")
    should_recommend_products: bool = Field(default=False, description="是否需要继续推荐商城商品。")
    product_keyword: str | None = Field(default=None, description="商品搜索关键词。")
    product_usage: str | None = Field(default=None, description="商品搜索适用场景。")

    @field_validator("diagnosis_text", "product_keyword", "product_usage")
    @classmethod
    def _normalize_text(cls, value: str | None) -> str | None:
        """
        功能描述：
            清理最终诊断文本字段的首尾空白。

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

    @model_validator(mode="after")
    def _validate_shape(self) -> "ConsultationFinalDiagnosisSchema":
        """
        功能描述：
            校验最终诊断节点输出形态。

        参数说明：
            无。

        返回值：
            ConsultationFinalDiagnosisSchema: 校验后的模型自身。

        异常说明：
            ValueError: 诊断文本为空，或商品推荐字段不完整时抛出。
        """

        if not self.diagnosis_text:
            raise ValueError("diagnosis_text 不能为空")
        if not self.should_recommend_products:
            self.product_keyword = None
            self.product_usage = None
            return self
        if not any([self.product_keyword, self.product_usage]):
            raise ValueError("should_recommend_products=true 时至少提供 product_keyword 或 product_usage")
        return self


__all__ = [
    "ConsultationFinalDiagnosisSchema",
    "ConsultationQuestionSchema",
]
