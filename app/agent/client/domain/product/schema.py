from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProductIdRequest(BaseModel):
    """按商品 ID 查询请求参数。"""

    model_config = ConfigDict(extra="forbid")

    product_id: int = Field(ge=1, description="商品 ID")


class ProductSearchRequest(BaseModel):
    """商品搜索请求参数。"""

    model_config = ConfigDict(extra="forbid")

    keyword: str | None = Field(default=None, description="搜索关键词")
    category_name: str | None = Field(default=None, description="商品分类名称")
    usage: str | None = Field(default=None, description="商品用途或适用场景")
    page_num: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(
        default=10,
        ge=1,
        le=20,
        description="每页数量，范围 1-20",
    )

    @field_validator("keyword", "category_name", "usage")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def _validate_query_present(self) -> "ProductSearchRequest":
        if not any([self.keyword, self.category_name, self.usage]):
            raise ValueError("keyword、category_name、usage 不能同时为空")
        return self


__all__ = [
    "ProductIdRequest",
    "ProductSearchRequest",
]
