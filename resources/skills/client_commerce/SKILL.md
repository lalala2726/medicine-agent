---
name: client_commerce
description: 客户端 commerce 节点工具总目录，覆盖商品、订单与售后问题，并提供 references 资源索引。
---

# Client Commerce Skill

## 使用原则

1. 这个 skill 是 `commerce_agent` 的总目录，先用它确定应该查订单、商品还是售后。
2. 默认只有基础工具 `list_loadable_tools`、`load_tools` 可直接使用；业务工具必须先通过 `load_tools` 加载。
3. 如果不确定精确工具名，先调用 `list_loadable_tools` 查看完整工具目录。
4. `load_tools` 支持一次同时加载多个工具；如果一个问题要同时查订单和售后，就把多个工具名一起放进 `tool_keys`。
5. `load_tools` 是你自己的内部加载步骤，不需要等待用户确认。
6. 需要精确字段、嵌套结构、动作工具参数或返回值解释时，再读取 `references/ORDER.md`、`references/PRODUCT.md`、
   `references/AFTER_SALE.md`。
7. Python 工具最终返回给模型的是业务 `data`，不是 Java 原始 `AjaxResult` 外层壳，所以不要期待 `code`、`message`、`timestamp`
   这些外层字段。
8. `open_user_order_list` 和 `open_user_after_sale_list` 这类动作工具返回的是一段给用户看的确认文案，同时会向前端下发页面跳转动作。
9. 商品搜索类工具返回分页结构，通常包含 `total`、`pageNum`、`pageSize`、`rows`。
10. 订单详情、物流、售后详情、资格校验这类工具返回单个对象，不带分页壳。

## 基础工具

工具：

- `list_loadable_tools`
- `load_tools`

能拿到什么：

- 当前可加载的精确工具名目录，并按 `order / product / after_sale` 分组
- 通过一次调用同时加载多个业务工具，供当前 `commerce_agent` 后续继续调用

## 订单与履约工具

这些工具统一归属 `commerce_agent`，默认不可见，需要先通过 `load_tools` 加载。

工具：

- `open_user_order_list`
- `get_order_detail`
- `get_order_shipping`
- `get_order_timeline`
- `check_order_cancelable`

能拿到什么：

- 打开订单列表页面，可选按订单状态筛选
- 订单金额、支付方式、收货信息、商品明细、物流摘要
- 发货状态、物流公司、运单号、轨迹节点
- 订单从创建到当前状态的时间线
- 当前订单是否允许取消，以及不能取消的原因

完整说明：

- [references/ORDER.md](references/ORDER.md)

## 商品与说明书工具

这些工具统一归属 `commerce_agent`，默认不可见，需要先通过 `load_tools` 加载。

工具：

- `search_products`
- `get_product_detail`
- `get_product_spec`

能拿到什么：

- 商品搜索结果、分页信息、商品 ID、商品名称、封面图、价格
- 商品详情、分类、单位、库存、销量、图片、药品说明信息
- 规格属性、通用名、成分、包装、有效期、用法用量、禁忌、说明书全文

完整说明：

- [references/PRODUCT.md](references/PRODUCT.md)

## 售后与资格工具

这些工具统一归属 `commerce_agent`，默认不可见，需要先通过 `load_tools` 加载。

工具：

- `open_user_after_sale_list`
- `get_after_sale_detail`
- `check_after_sale_eligibility`

能拿到什么：

- 打开售后列表页面，可选按售后状态筛选
- 售后单状态、退款金额、驳回原因、凭证图片、处理时间线
- 某笔订单或某个订单项是否还能申请退款/退货/换货，以及可退款金额

完整说明：

- [references/AFTER_SALE.md](references/AFTER_SALE.md)

## 何时继续追问

1. 用户只说“查一下我的订单”但没有要求打开订单列表，也没有给订单号时，先问清是想打开列表还是查询某一笔订单。
2. 用户说“能不能退”但没有订单号时，先要订单号；如果用户是在问某个订单项，优先再要 `orderItemId`。
3. 用户说“这个药适不适合我”但问题里混有明显症状判断，商品信息可以回答，但医学诊断不要在本 skill 内臆断。
