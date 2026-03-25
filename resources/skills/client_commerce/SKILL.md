---
name: client_commerce
description: 客户端 commerce 节点工具总目录，覆盖商品、订单与售后问题，并提供 references 资源索引。
---

# Client Commerce Skill

## 使用原则

1. 这个 skill 是 `commerce_agent` 的总目录，先用它确定应该查订单、商品还是售后。
2. 需要精确字段、嵌套结构、动作工具参数或返回值解释时，再读取 `references/ORDER.md`、`references/PRODUCT.md`、
   `references/AFTER_SALE.md`。
3. Python 工具最终返回给模型的是业务 `data`，不是 Java 原始 `AjaxResult` 外层壳，所以不要期待 `code`、`message`、`timestamp`
   这些外层字段。
4. `open_user_order_list` 和 `open_user_after_sale_list` 这类动作工具返回的是一段给用户看的确认文案，同时会向前端下发页面跳转动作。
5. 商品搜索类工具返回分页结构，通常包含 `total`、`pageNum`、`pageSize`、`rows`。
6. 订单详情、物流、售后详情、资格校验这类工具返回单个对象，不带分页壳。

## 订单与履约工具

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
