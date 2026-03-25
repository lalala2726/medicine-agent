---
name: admin_tools
description: 管理端单 Agent 可申请的业务工具目录，包含工具名称、适用场景、关键参数与常见用途。
license: Apache-2.0
metadata:
  author: Chuang
  version: "1.0"
---

# Admin Tools Skill

## 使用原则

1. 当前全部业务工具都允许直接调用，无需先申请权限。
2. `request_tool_access` 仍可用，但仅用于记录申请动作或确认工具目录，不再是权限开关。
3. 工具 key 统一为 `snake_case`，调用时必须使用下面列出的精确名称。
4. 同一个问题只调用最小必要工具，避免无关查询。

## 基础工具

| 工具名                        | 作用              | 何时使用                             |
|----------------------------|-----------------|----------------------------------|
| `request_tool_access`      | 记录工具申请动作并确认工具目录 | 不确定工具名、想确认工具目录时                  |
| `search_knowledge_context` | 检索知识库内容         | 回答制度说明、字段解释、规则定义、FAQ、配置说明等知识型问题时 |
| `get_safe_user_info`       | 获取当前登录用户的基础信息   | 需要确认当前操作人身份、用户名、昵称等上下文时          |

## 订单工具

| 工具名              | 作用          | 关键参数                                                             |
|------------------|-------------|------------------------------------------------------------------|
| `order_list`     | 查询订单列表      | `page_num`、`page_size`、`order_no`、`order_status`、`receiver_name` |
| `order_detail`   | 查询一个或多个订单详情 | `order_id: list[str]`                                            |
| `order_timeline` | 查询订单流程时间线   | `order_id: int`                                                  |
| `order_shipping` | 查询订单发货记录    | `order_id: int`                                                  |

适用场景：

- 查某段时间或某状态下的订单
- 看订单明细、收货信息、商品明细
- 看订单流转节点或发货物流信息

## 商品工具

| 工具名              | 作用             | 关键参数                                                                         |
|------------------|----------------|------------------------------------------------------------------------------|
| `product_list`   | 查询商品列表         | `page_num`、`page_size`、`name`、`category_id`、`status`、`min_price`、`max_price` |
| `product_detail` | 查询商品详情         | `product_id: list[str]`                                                      |
| `drug_detail`    | 查询药品说明书与适应症等详情 | `product_id: list[str]`                                                      |

适用场景：

- 筛选商品范围
- 查看商品价格、库存、状态
- 查询药品说明书、适应症、用法用量

## 售后工具

| 工具名                 | 作用       | 关键参数                                                                              |
|---------------------|----------|-----------------------------------------------------------------------------------|
| `after_sale_list`   | 查询售后申请列表 | `page_num`、`page_size`、`after_sale_type`、`after_sale_status`、`order_no`、`user_id` |
| `after_sale_detail` | 查询售后详情   | `after_sale_id: int`                                                              |

适用场景：

- 看待处理售后单
- 定位某订单或某用户的售后情况
- 查看售后处理进度和结果

## 用户工具

| 工具名                 | 作用       | 关键参数                                                               |
|---------------------|----------|--------------------------------------------------------------------|
| `user_list`         | 查询用户列表   | `page_num`、`page_size`、`id`、`username`、`nickname`、`roles`、`status` |
| `user_detail`       | 查询用户详情   | `user_id: int`                                                     |
| `user_wallet`       | 查询用户钱包信息 | `user_id: int`                                                     |
| `user_wallet_flow`  | 查询用户钱包流水 | `user_id`、`page_num`、`page_size`                                   |
| `user_consume_info` | 查询用户消费记录 | `user_id`、`page_num`、`page_size`                                   |

适用场景：

- 定位用户
- 看用户详情、角色、状态
- 查钱包余额、流水和消费信息

## 运营分析工具

| 工具名                                        | 作用           | 关键参数           |
|--------------------------------------------|--------------|----------------|
| `analytics_realtime_overview`              | 查询实时运营总览     | 无              |
| `analytics_range_summary`                  | 查询经营结果汇总     | `days`         |
| `analytics_conversion_summary`             | 查询支付转化汇总     | `days`         |
| `analytics_fulfillment_summary`            | 查询履约时效汇总     | `days`         |
| `analytics_after_sale_efficiency_summary`  | 查询售后处理时效汇总   | `days`         |
| `analytics_after_sale_status_distribution` | 查询售后状态分布     | `days`         |
| `analytics_after_sale_reason_distribution` | 查询售后原因分布     | `days`         |
| `analytics_top_selling_products`           | 查询热销商品排行     | `days`、`limit` |
| `analytics_return_refund_risk_products`    | 查询退货退款风险商品排行 | `days`、`limit` |
| `analytics_sales_trend`                    | 查询成交趋势       | `days`         |
| `analytics_after_sale_trend`               | 查询售后趋势       | `days`         |

适用场景：

- 看实时运营指标
- 看支付、履约、售后时效
- 看趋势、分布、排行
- 为图表展示准备数据

## 使用示例

1. 用户要查订单列表：
   直接调用 `order_list`
2. 用户要看订单详情和物流：
   直接调用 `order_detail`、`order_shipping`
3. 用户要查用户钱包和消费：
   直接调用 `user_wallet`、`user_consume_info`
4. 用户要看成交趋势并给图表建议：
   直接调用 `analytics_sales_trend`，再结合 `chart` skill 给出图表建议
