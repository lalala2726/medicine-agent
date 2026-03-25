from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProductIdRequest(BaseModel):
    """
    功能描述：
        按商品 ID 查询的请求参数模型。

    参数说明：
        product_id (int): 商品 ID。

    返回值：
        无（数据模型定义）。

    异常说明：
        无。
    """

    model_config = ConfigDict(extra="forbid")

    product_id: int = Field(ge=1, description="商品 ID")


class ProductSearchRequest(BaseModel):
    """
    功能描述：
        商品搜索请求参数模型。

    参数说明：
        keyword (str | None): 搜索关键词。
        category_name (str | None): 商品分类名称。
        usage (str | None): 商品用途或适用场景。
        page_num (int): 页码。
        page_size (int): 每页数量。

    返回值：
        无（数据模型定义）。

    异常说明：
        ValueError: 当搜索条件全部为空时抛出。
    """

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
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """
        功能描述：
            规范化可选文本查询条件。

        参数说明：
            value (str | None): 原始文本条件。

        返回值：
            str | None: 去空白后的文本；空串归一化为 `None`。

        异常说明：
            无。
        """

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_query_present(self) -> "ProductSearchRequest":
        """
        功能描述：
            校验至少存在一个有效搜索条件。

        参数说明：
            无。

        返回值：
            ProductSearchRequest: 当前模型实例。

        异常说明：
            ValueError: 当搜索条件全部为空时抛出。
        """

        if not any([self.keyword, self.category_name, self.usage]):
            raise ValueError("keyword、category_name、usage 不能同时为空")
        return self


__all__ = [
    "ProductIdRequest",
    "ProductSearchRequest",
]
