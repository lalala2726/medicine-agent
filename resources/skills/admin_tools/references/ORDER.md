# Order Tools Reference

## 领域说明

订单领域工具用于查询订单列表、订单详情、订单流程和物流信息。  
返回结构以 Python 工具实际返回的 `data` 为准，不包含 Java 原始响应外层的 `code/message/timestamp`。

## 工具清单

| 工具名              | 适用场景                | 关键参数                                                                                                         |
|------------------|---------------------|--------------------------------------------------------------------------------------------------------------|
| `order_list`     | 查订单范围、筛选订单、做订单汇总前取样 | `page_num`、`page_size`、`order_no`、`pay_type`、`order_status`、`delivery_type`、`receiver_name`、`receiver_phone` |
| `order_detail`   | 查看订单完整明细            | `order_id: list[str]`                                                                                        |
| `order_timeline` | 查看订单状态推进过程          | `order_id: int`                                                                                              |
| `order_shipping` | 查看发货和物流轨迹           | `order_id: int`                                                                                              |

## order_list

### 实际返回结构

分页对象：

- `total`
- `pageNum`
- `pageSize`
- `rows`

`rows[]` 中每一项是订单列表对象，通常包含：

- `id`：订单 ID
- `orderNo`：订单编号
- `totalAmount`：订单总金额
- `payType`：支付方式编码
- `orderStatus`：订单状态编码
- `payTime`：支付时间
- `createTime`：创建时间
- `productInfo`：首个商品信息对象

`productInfo` 常见字段：

- `productName`：商品名称
- `productImage`：商品图片
- `productPrice`：商品价格
- `productCategory`：商品分类
- `productId`：商品 ID
- `quantity`：商品数量

### 关键字段解释

- 这是订单分页列表，不是完整详情。
- `productInfo` 是订单列表里携带的首个商品摘要，适合快速浏览，不等于完整商品明细数组。
- `payType`、`orderStatus` 是业务编码字段。

### 常见可回答问题

- 某个状态下有多少订单
- 某个收货人或订单号相关的订单有哪些
- 某段时间最近创建或支付的订单概览

## order_detail

### 实际返回结构

返回订单详情数组 `list[object]`，每项通常包含：

- `userInfo`
- `deliveryInfo`
- `orderInfo`
- `productInfo`

`userInfo` 常见字段：

- `userId`
- `nickname`
- `phoneNumber`

`deliveryInfo` 常见字段：

- `receiverName`
- `receiverAddress`
- `receiverPhone`
- `deliveryMethod`

`orderInfo` 常见字段：

- `orderNo`
- `orderStatus`
- `payType`
- `totalAmount`
- `payAmount`
- `freightAmount`

`productInfo[]` 中每项常见字段：

- `productId`
- `productName`
- `productImage`
- `productPrice`
- `productQuantity`
- `productTotalAmount`

### 关键字段解释

- 这是完整订单详情数组，即使只查一个订单，也要按数组理解。
- `productInfo` 是商品明细数组，不是订单列表里的单个摘要对象。
- `orderStatus`、`payType` 仍然是编码字段。

### 常见可回答问题

- 某订单的收货地址和联系人是谁
- 某订单买了哪些商品、数量多少、金额多少
- 某订单实际支付金额和运费是多少

## order_timeline

### 实际返回结构

返回订单时间线数组 `list[object]`，每项常见字段：

- `id`
- `orderId`
- `eventType`
- `eventStatus`
- `operatorType`
- `description`
- `createdTime`

### 关键字段解释

- `eventType`：事件类型，例如创建、支付、发货、完成等流程节点类型。
- `eventStatus`：事件状态。
- `operatorType`：操作方类型，常见是 `USER`、`ADMIN`、`SYSTEM`。
- `description`：该节点的人类可读描述。

### 常见可回答问题

- 订单当前推进到了哪一步
- 最近一次状态变化是什么
- 某个节点是用户触发、后台触发还是系统触发

## order_shipping

### 实际返回结构

返回单个物流对象，常见字段：

- `orderId`
- `orderNo`
- `orderStatus`
- `orderStatusName`
- `logisticsCompany`
- `trackingNumber`
- `shipmentNote`
- `deliverTime`
- `receiveTime`
- `status`
- `statusName`
- `receiverInfo`
- `nodes`

`receiverInfo` 常见字段：

- `receiverName`
- `receiverPhone`
- `receiverDetail`
- `deliveryType`
- `deliveryTypeName`

`nodes[]` 常见字段：

- `time`
- `content`
- `location`

### 关键字段解释

- `orderStatus` / `orderStatusName` 是订单层状态。
- `status` / `statusName` 是物流层状态。
- `nodes` 是物流轨迹节点数组，适合直接整理成时间线。

### 常见可回答问题

- 订单是否已发货、何时发货
- 使用了哪家物流公司、物流单号是什么
- 当前物流走到了哪里、最近一条轨迹是什么
