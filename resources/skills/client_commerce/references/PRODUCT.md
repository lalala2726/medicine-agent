# Product References

## search_products

适用场景：

- 用户想找某类商品
- 用户想按用途、关键词或分类挑选商品
- 用户要对比几个商品前，先需要缩小候选范围

关键参数：

- `keyword`：可选，商品关键词
- `category_name`：可选，分类名称
- `usage`：可选，用途或适用场景
- `page_num`：可选，页码，默认 1
- `page_size`：可选，每页条数，默认 10

实际返回结构：

- 分页字段：
    - `total`
    - `pageNum`
    - `pageSize`
    - `rows`
- `rows[]`：
    - `productId`
    - `productName`
    - `cover`
    - `price`

常见可回答问题：

- “有没有止咳的商品”
- “给我找点感冒药”
- “有哪些适合家里常备的退烧药”

继续追问规则：

- 用户需求太泛时，先用搜索工具；必要时再追问更细条件
- 用户已经给出明确商品 ID 时，不要先搜，直接查详情或规格

## get_product_detail

适用场景：

- 用户已经给出商品 ID，想看价格、库存、分类、图片、销量、药品说明信息

关键参数：

- `product_id`：必填，商品 ID

实际返回结构：

- 顶层字段：
    - `id`
    - `name`
    - `categoryId`
    - `categoryName`
    - `unit`
    - `price`
    - `stock`
    - `status`
    - `deliveryType`
    - `sales`
    - `images`
    - `drugDetail`
- `images[]`：商品图片 URL 列表
- `drugDetail`：
    - `commonName`
    - `composition`
    - `characteristics`
    - `packaging`
    - `validityPeriod`
    - `storageConditions`
    - `productionUnit`
    - `approvalNumber`
    - `executiveStandard`
    - `originType`
    - `isOutpatientMedicine`
    - `warmTips`
    - `brand`
    - `prescription`
    - `efficacy`
    - `usageMethod`
    - `adverseReactions`
    - `precautions`
    - `taboo`
    - `instruction`

关键解释：

- `status` 是商品状态编码
- `deliveryType` 是配送方式编码
- `prescription` 用于判断是否处方药
- `instruction` 是药品说明书全文

常见可回答问题：

- “这个商品多少钱，还有库存吗”
- “这个药是处方药吗”
- “这个商品的功效和注意事项是什么”

## get_product_spec

适用场景：

- 用户已经给出商品 ID，想更聚焦地看规格属性、成分、包装、有效期、禁忌、说明书等字段

关键参数：

- `product_id`：必填，商品 ID

实际返回结构：

- `productId`
- `productName`
- `categoryName`
- `unit`
- `commonName`
- `composition`
- `characteristics`
- `packaging`
- `validityPeriod`
- `storageConditions`
- `productionUnit`
- `approvalNumber`
- `executiveStandard`
- `originType`
- `brand`
- `prescription`
- `efficacy`
- `usageMethod`
- `adverseReactions`
- `precautions`
- `taboo`
- `instruction`

常见可回答问题：

- “这个药怎么吃”
- “这个药的成分和禁忌是什么”
- “包装规格和有效期是多少”
- “请把说明书重点告诉我”

继续追问规则：

- 用户只给商品名称、没有商品 ID 时，先用 `search_products`
- 用户在问适用性时，如果问题本质是医学诊断，不要把商品信息硬说成诊断结论
