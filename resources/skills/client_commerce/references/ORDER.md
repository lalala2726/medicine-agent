# Order References

本文件描述 `commerce_agent` 下订单与履约工具的精确工具名、入参与返回结构。

## open_user_order_list

适用场景：

- 用户明确要求打开、进入、查看订单列表。
- 用户要求查看待支付、待发货、待收货、已完成、已取消等特定状态的订单列表。

关键参数：

- `orderStatus`
    - 可选
    - 允许值：`PENDING_PAYMENT`、`PENDING_SHIPMENT`、`PENDING_RECEIPT`、`COMPLETED`、`CANCELLED`
    - 用户没有明确指定状态时不要传

返回值：

- 返回一段给用户看的确认文案，例如“已为你打开订单列表”“已为你打开待发货订单列表”
- 同时向前端下发页面跳转动作：
    - `target = user_order_list`
    - `payload = {"orderStatus": ...}`

常见可回答问题：

- “打开我的订单”
- “帮我看待发货订单”
- “我想进已完成订单列表”

## get_order_detail

适用场景：

- 用户已经提供订单号，想查订单详情、金额、商品、收货信息、支付状态、物流摘要

关键参数：

- `order_no`：必填，订单编号

实际返回结构：

- 顶层字段：
    - `id`
    - `orderNo`
    - `orderStatus`
    - `orderStatusName`
    - `totalAmount`
    - `payAmount`
    - `freightAmount`
    - `payType`
    - `payTypeName`
    - `deliveryType`
    - `deliveryTypeName`
    - `paid`
    - `payExpireTime`
    - `payTime`
    - `deliverTime`
    - `receiveTime`
    - `finishTime`
    - `createTime`
    - `note`
    - `afterSaleFlag`
    - `afterSaleFlagName`
    - `refundStatus`
    - `refundPrice`
    - `refundTime`
    - `receiverInfo`
    - `items`
    - `shippingInfo`
- `receiverInfo`：
    - `receiverName`
    - `receiverPhone`
    - `receiverDetail`
- `items[]`：
    - `id`
    - `productId`
    - `productName`
    - `imageUrl`
    - `quantity`
    - `price`
    - `totalPrice`
    - `afterSaleStatus`
    - `afterSaleStatusName`
    - `refundedAmount`
- `shippingInfo`：
    - `logisticsCompany`
    - `trackingNumber`
    - `shippingStatus`
    - `shippingStatusName`
    - `shipTime`

关键解释：

- `paid` 表示是否已支付
- `afterSaleFlag` / `afterSaleFlagName` 表示整单售后标记
- `refundStatus` / `refundPrice` / `refundTime` 用于判断当前订单是否发生过退款及退款金额

常见可回答问题：

- “订单 O2025xxx 现在是什么状态”
- “这单一共多少钱，实付多少”
- “这单买了哪些商品”
- “帮我看一下收货地址和支付方式”

## get_order_shipping

适用场景：

- 用户已经提供订单号，想确认是否发货、物流公司、运单号、最新物流轨迹

关键参数：

- `order_no`：必填，订单编号

实际返回结构：

- 顶层字段：
    - `orderId`
    - `orderNo`
    - `orderStatus`
    - `orderStatusName`
    - `shippingStatus`
    - `shippingStatusName`
    - `logisticsCompany`
    - `trackingNumber`
    - `shipmentNote`
    - `deliverTime`
    - `receiveTime`
    - `receiverInfo`
    - `nodes`
- `receiverInfo`：
    - `receiverName`
    - `receiverPhone`
    - `receiverDetail`
    - `deliveryType`
    - `deliveryTypeName`
- `nodes[]`：
    - `time`
    - `content`
    - `location`

常见可回答问题：

- “物流到哪了”
- “什么时候发货的”
- “快递单号是多少”
- “最近一条物流更新是什么”

## get_order_timeline

适用场景：

- 用户已经提供订单号，想看订单从创建到当前状态的过程节点

关键参数：

- `order_no`：必填，订单编号

实际返回结构：

- 顶层字段：
    - `orderId`
    - `orderNo`
    - `orderStatus`
    - `orderStatusName`
    - `timeline`
- `timeline[]`：
    - `id`
    - `eventType`
    - `eventTypeName`
    - `eventStatus`
    - `eventStatusName`
    - `operatorType`
    - `operatorTypeName`
    - `description`
    - `createdTime`

常见可回答问题：

- “这单经历了哪些步骤”
- “订单什么时候支付、发货、完成的”
- “这个状态是谁触发的”

## check_order_cancelable

适用场景：

- 用户已经提供订单号，想确认当前订单能不能取消，以及不能取消的原因

关键参数：

- `order_no`：必填，订单编号

实际返回结构：

- `orderNo`
- `orderStatus`
- `orderStatusName`
- `cancelable`
- `reasonCode`
- `reasonMessage`

常见可回答问题：

- “这单还能取消吗”
- “为什么不能取消”

继续追问规则：

- 用户没有提供订单号时，不要直接调用详情、物流、时间线、取消校验工具，先要订单号
- 用户只是想打开列表而不是查询某一笔订单时，优先使用 `open_user_order_list`
