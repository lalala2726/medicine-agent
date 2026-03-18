# Client Agent API 文档

本文档基于当前仓库实际代码扫描整理，扫描范围如下：

- `medicine-agent/src/main/java/cn/zhangchuangla/medicine/agent/controller/client/AgentClientOrderController.java`
- `medicine-agent/src/main/java/cn/zhangchuangla/medicine/agent/controller/client/AgentClientProductController.java`
- `medicine-agent/src/main/java/cn/zhangchuangla/medicine/agent/controller/client/AgentClientAfterSaleController.java`

适用对象：AI 端 / 智能体端调用客户端工具接口时参考。

## 1. 通用约定

### 1.1 基础路径

所有接口统一前缀：

```text
/agent/client
```

当前共 9 个接口，全部为 `GET` 接口。

### 1.2 登录态与用户范围

- 订单、售后数据不允许前端自行传 `userId`
- 服务端从 Spring Security 上下文中自动获取当前登录用户
- 未登录时通常返回：

```json
{
  "code": 401,
  "message": "用户未登录",
  "timestamp": 1710000000000,
  "data": null
}
```

### 1.3 请求头

这些 controller 都打了 `@InternalAgentHeaderTrace`，当前会记录以下请求头到日志：

- `X-Agent-Key`
- `X-Agent-Timestamp`
- `X-Agent-Nonce`
- `X-Forwarded-For`

注意：

- 目前代码里是“记录日志”，不是强制校验拦截
- AI 端如果已有这些头，建议继续透传

### 1.4 统一响应结构

所有接口统一返回 `AjaxResult<T>`：

```json
{
  "code": 200,
  "message": "操作成功",
  "timestamp": 1710000000000,
  "data": {}
}
```

常见 `code`：

- `200`：成功
- `400`：参数错误 / 业务校验失败
- `401`：未登录
- `204`：查询为空

注意：

- 某些失败场景 HTTP 状态码仍可能是 `200`
- AI 端请以响应体里的 `code` 作为最终判断标准

### 1.5 分页响应结构

分页接口返回 `AjaxResult<TableDataResult>`，其 `data` 结构如下：

```json
{
  "total": 1,
  "pageNum": 1,
  "pageSize": 10,
  "rows": []
}
```

### 1.6 meta 字段

agent 层会自动为非空响应追加 `data.meta`：

```json
{
  "entityDescription": "客户端智能体商品搜索结果",
  "fieldDescriptions": {
    "productName": "商品名称"
  }
}
```

AI 端可以直接忽略该字段。

### 1.7 编码字段序列化规则

带 `@AgentCodeLabel` 的字段，实际返回不是原始字符串，而是对象：

```json
"orderStatus": {
"value": "PENDING_PAYMENT",
"description": "待支付"
}
```

因此 AI 端处理时建议：

- 优先读 `xxxName`
- 如果没有 `xxxName`，则读编码对象里的 `description`
- 不要假设 `orderStatus`、`afterSaleStatus`、`shippingStatus` 这类字段一定是纯字符串

### 1.8 脱敏字段

以下字段会脱敏返回：

- `receiverPhone`

## 2. 接口总览

### 2.1 订单

- `GET /agent/client/order/{orderNo}` 获取订单详情
- `GET /agent/client/order/shipping/{orderNo}` 获取订单物流
- `GET /agent/client/order/timeline/{orderNo}` 获取订单时间线
- `GET /agent/client/order/cancel-check/{orderNo}` 校验是否可取消订单

### 2.2 商品

- `GET /agent/client/product/search` 搜索商品
- `GET /agent/client/product/{productId}` 获取商品详情
- `GET /agent/client/product/spec/{productId}` 获取商品规格属性

### 2.3 售后

- `GET /agent/client/after-sale/{afterSaleNo}` 获取售后详情
- `GET /agent/client/after-sale/eligibility` 校验售后资格

## 3. 订单接口

---

## 3.1 获取订单详情

### 路径

```text
GET /agent/client/order/{orderNo}
```

### 路径参数

| 参数名       | 类型     | 必填 | 说明   |
|-----------|--------|----|------|
| `orderNo` | string | 是  | 订单编号 |

### 返回 data 结构

| 字段                  | 类型       | 说明                                   |
|---------------------|----------|--------------------------------------|
| `id`                | long     | 订单 ID                                |
| `orderNo`           | string   | 订单编号                                 |
| `orderStatus`       | object   | 订单状态编码对象，格式 `{ value, description }` |
| `orderStatusName`   | string   | 订单状态名称                               |
| `totalAmount`       | number   | 订单总金额                                |
| `payAmount`         | number   | 实际支付金额                               |
| `freightAmount`     | number   | 运费金额                                 |
| `payType`           | object   | 支付方式编码对象，格式 `{ value, description }` |
| `payTypeName`       | string   | 支付方式名称                               |
| `deliveryType`      | string   | 配送方式编码                               |
| `deliveryTypeName`  | string   | 配送方式名称                               |
| `paid`              | integer  | 是否支付，`0/1`                           |
| `payExpireTime`     | datetime | 支付过期时间                               |
| `payTime`           | datetime | 支付时间                                 |
| `deliverTime`       | datetime | 发货时间                                 |
| `receiveTime`       | datetime | 确认收货时间                               |
| `finishTime`        | datetime | 完成时间                                 |
| `createTime`        | datetime | 下单时间                                 |
| `note`              | string   | 用户留言                                 |
| `afterSaleFlag`     | string   | 订单售后标记编码                             |
| `afterSaleFlagName` | string   | 订单售后标记名称                             |
| `refundStatus`      | string   | 退款状态                                 |
| `refundPrice`       | number   | 退款金额                                 |
| `refundTime`        | datetime | 退款时间                                 |
| `receiverInfo`      | object   | 收货人信息                                |
| `items`             | array    | 订单商品列表                               |
| `shippingInfo`      | object   | 物流摘要信息                               |

#### receiverInfo

| 字段               | 类型     | 说明          |
|------------------|--------|-------------|
| `receiverName`   | string | 收货人姓名       |
| `receiverPhone`  | string | 收货人手机号，脱敏返回 |
| `receiverDetail` | string | 收货详细地址      |

#### items[]

| 字段                    | 类型      | 说明        |
|-----------------------|---------|-----------|
| `id`                  | long    | 订单项 ID    |
| `productId`           | long    | 商品 ID     |
| `productName`         | string  | 商品名称      |
| `imageUrl`            | string  | 商品图片      |
| `quantity`            | integer | 购买数量      |
| `price`               | number  | 商品单价      |
| `totalPrice`          | number  | 小计金额      |
| `afterSaleStatus`     | string  | 订单项售后状态编码 |
| `afterSaleStatusName` | string  | 订单项售后状态名称 |
| `refundedAmount`      | number  | 已退款金额     |

#### shippingInfo

| 字段                   | 类型       | 说明     |
|----------------------|----------|--------|
| `logisticsCompany`   | string   | 物流公司   |
| `trackingNumber`     | string   | 物流单号   |
| `shippingStatus`     | string   | 物流状态编码 |
| `shippingStatusName` | string   | 物流状态名称 |
| `shipTime`           | datetime | 发货时间   |

---

## 3.2 获取订单物流

### 路径

```text
GET /agent/client/order/shipping/{orderNo}
```

### 路径参数

| 参数名       | 类型     | 必填 | 说明   |
|-----------|--------|----|------|
| `orderNo` | string | 是  | 订单编号 |

### 返回 data 结构

| 字段                   | 类型       | 说明       |
|----------------------|----------|----------|
| `orderId`            | long     | 订单 ID    |
| `orderNo`            | string   | 订单编号     |
| `orderStatus`        | object   | 订单状态编码对象 |
| `orderStatusName`    | string   | 订单状态名称   |
| `shippingStatus`     | object   | 物流状态编码对象 |
| `shippingStatusName` | string   | 物流状态名称   |
| `logisticsCompany`   | string   | 物流公司     |
| `trackingNumber`     | string   | 运单号      |
| `shipmentNote`       | string   | 发货备注     |
| `deliverTime`        | datetime | 发货时间     |
| `receiveTime`        | datetime | 签收时间     |
| `receiverInfo`       | object   | 收货人信息    |
| `nodes`              | array    | 物流轨迹节点   |

#### receiverInfo

| 字段                 | 类型     | 说明          |
|--------------------|--------|-------------|
| `receiverName`     | string | 收货人姓名       |
| `receiverPhone`    | string | 收货人手机号，脱敏返回 |
| `receiverDetail`   | string | 收货详细地址      |
| `deliveryType`     | string | 配送方式编码      |
| `deliveryTypeName` | string | 配送方式名称      |

#### nodes[]

| 字段         | 类型     | 说明   |
|------------|--------|------|
| `time`     | string | 节点时间 |
| `content`  | string | 节点内容 |
| `location` | string | 节点位置 |

---

## 3.3 获取订单时间线

### 路径

```text
GET /agent/client/order/timeline/{orderNo}
```

### 路径参数

| 参数名       | 类型     | 必填 | 说明   |
|-----------|--------|----|------|
| `orderNo` | string | 是  | 订单编号 |

### 返回 data 结构

| 字段                | 类型     | 说明         |
|-------------------|--------|------------|
| `orderId`         | long   | 订单 ID      |
| `orderNo`         | string | 订单编号       |
| `orderStatus`     | object | 当前订单状态编码对象 |
| `orderStatusName` | string | 当前订单状态名称   |
| `timeline`        | array  | 时间线节点      |

#### timeline[]

| 字段                 | 类型       | 说明        |
|--------------------|----------|-----------|
| `id`               | long     | 时间线 ID    |
| `eventType`        | object   | 事件类型编码对象  |
| `eventTypeName`    | string   | 事件类型名称    |
| `eventStatus`      | object   | 事件状态编码对象  |
| `eventStatusName`  | string   | 事件状态名称    |
| `operatorType`     | object   | 操作方类型编码对象 |
| `operatorTypeName` | string   | 操作方类型名称   |
| `description`      | string   | 节点描述      |
| `createdTime`      | datetime | 节点时间      |

---

## 3.4 校验是否可取消订单

### 路径

```text
GET /agent/client/order/cancel-check/{orderNo}
```

### 路径参数

| 参数名       | 类型     | 必填 | 说明   |
|-----------|--------|----|------|
| `orderNo` | string | 是  | 订单编号 |

### 返回 data 结构

| 字段                | 类型      | 说明         |
|-------------------|---------|------------|
| `orderNo`         | string  | 订单编号       |
| `orderStatus`     | object  | 当前订单状态编码对象 |
| `orderStatusName` | string  | 当前订单状态名称   |
| `cancelable`      | boolean | 是否允许取消     |
| `reasonCode`      | string  | 结果编码       |
| `reasonMessage`   | string  | 结果说明       |

### 当前 reasonCode

| code                          | 含义            |
|-------------------------------|---------------|
| `CAN_CANCEL`                  | 当前允许取消        |
| `ORDER_NOT_FOUND`             | 订单不存在或不属于当前用户 |
| `ORDER_STATUS_INVALID`        | 订单状态异常        |
| `ORDER_STATUS_NOT_CANCELABLE` | 当前状态不允许取消     |

## 4. 商品接口

---

## 4.1 搜索商品

### 路径

```text
GET /agent/client/product/search
```

### 查询参数

| 参数名            | 类型      | 必填 | 说明                   |
|----------------|---------|----|----------------------|
| `keyword`      | string  | 否  | 搜索关键词                |
| `categoryName` | string  | 否  | 商品分类名称               |
| `usage`        | string  | 否  | 商品用途/适用场景            |
| `pageNum`      | integer | 否  | 页码，默认 `1`            |
| `pageSize`     | integer | 否  | 每页数量，默认 `10`，最大 `20` |

### 校验规则

- `keyword`、`categoryName`、`usage` 不能同时为空
- `pageSize <= 20`

### 当前检索实现

- 当前实现是 ES 检索，不是数据库直接模糊查询

### 返回 data 结构

| 字段         | 类型    | 说明   |
|------------|-------|------|
| `total`    | long  | 总记录数 |
| `pageNum`  | long  | 当前页码 |
| `pageSize` | long  | 每页条数 |
| `rows`     | array | 商品列表 |

#### rows[]

| 字段            | 类型     | 说明    |
|---------------|--------|-------|
| `productId`   | long   | 商品 ID |
| `productName` | string | 商品名称  |
| `cover`       | string | 商品封面图 |
| `price`       | number | 商品价格  |

---

## 4.2 获取商品详情

### 路径

```text
GET /agent/client/product/{productId}
```

### 路径参数

| 参数名         | 类型   | 必填 | 说明               |
|-------------|------|----|------------------|
| `productId` | long | 是  | 商品 ID，且必须 `>= 1` |

### 返回 data 结构

| 字段             | 类型      | 说明                               |
|----------------|---------|----------------------------------|
| `id`           | long    | 商品 ID                            |
| `name`         | string  | 商品名称                             |
| `categoryId`   | long    | 商品分类 ID                          |
| `categoryName` | string  | 商品分类名称                           |
| `unit`         | string  | 商品单位                             |
| `price`        | number  | 商品价格                             |
| `stock`        | integer | 库存                               |
| `status`       | object  | 商品状态编码对象，`value` 当前为 `0/1`       |
| `deliveryType` | object  | 配送方式编码对象，`value` 当前为 legacy 整型编码 |
| `sales`        | integer | 销量                               |
| `images`       | array   | 商品图片列表                           |
| `drugDetail`   | object  | 药品说明信息                           |

#### drugDetail

| 字段                     | 类型      | 说明      |
|------------------------|---------|---------|
| `commonName`           | string  | 药品通用名   |
| `composition`          | string  | 成分      |
| `characteristics`      | string  | 性状      |
| `packaging`            | string  | 包装规格    |
| `validityPeriod`       | string  | 有效期     |
| `storageConditions`    | string  | 贮藏条件    |
| `productionUnit`       | string  | 生产单位    |
| `approvalNumber`       | string  | 批准文号    |
| `executiveStandard`    | string  | 执行标准    |
| `originType`           | string  | 产地类型    |
| `isOutpatientMedicine` | boolean | 是否外用药   |
| `warmTips`             | string  | 温馨提示    |
| `brand`                | string  | 品牌      |
| `prescription`         | boolean | 是否处方药   |
| `efficacy`             | string  | 功能主治    |
| `usageMethod`          | string  | 用法用量    |
| `adverseReactions`     | string  | 不良反应    |
| `precautions`          | string  | 注意事项    |
| `taboo`                | string  | 禁忌      |
| `instruction`          | string  | 药品说明书全文 |

---

## 4.3 获取商品规格属性

### 路径

```text
GET /agent/client/product/spec/{productId}
```

### 路径参数

| 参数名         | 类型   | 必填 | 说明               |
|-------------|------|----|------------------|
| `productId` | long | 是  | 商品 ID，且必须 `>= 1` |

### 返回 data 结构

| 字段                  | 类型      | 说明      |
|---------------------|---------|---------|
| `productId`         | long    | 商品 ID   |
| `productName`       | string  | 商品名称    |
| `categoryName`      | string  | 商品分类名称  |
| `unit`              | string  | 商品单位    |
| `commonName`        | string  | 药品通用名   |
| `composition`       | string  | 成分      |
| `characteristics`   | string  | 性状      |
| `packaging`         | string  | 包装规格    |
| `validityPeriod`    | string  | 有效期     |
| `storageConditions` | string  | 贮藏条件    |
| `productionUnit`    | string  | 生产单位    |
| `approvalNumber`    | string  | 批准文号    |
| `executiveStandard` | string  | 执行标准    |
| `originType`        | string  | 产地类型    |
| `brand`             | string  | 品牌      |
| `prescription`      | boolean | 是否处方药   |
| `efficacy`          | string  | 功能主治    |
| `usageMethod`       | string  | 用法用量    |
| `adverseReactions`  | string  | 不良反应    |
| `precautions`       | string  | 注意事项    |
| `taboo`             | string  | 禁忌      |
| `instruction`       | string  | 药品说明书全文 |

## 5. 售后接口

---

## 5.1 获取售后详情

### 路径

```text
GET /agent/client/after-sale/{afterSaleNo}
```

### 路径参数

| 参数名           | 类型     | 必填 | 说明   |
|---------------|--------|----|------|
| `afterSaleNo` | string | 是  | 售后单号 |

### 返回 data 结构

| 字段                    | 类型       | 说明       |
|-----------------------|----------|----------|
| `id`                  | long     | 售后申请 ID  |
| `afterSaleNo`         | string   | 售后单号     |
| `orderId`             | long     | 订单 ID    |
| `orderNo`             | string   | 订单编号     |
| `orderItemId`         | long     | 订单项 ID   |
| `userId`              | long     | 用户 ID    |
| `userNickname`        | string   | 用户昵称     |
| `afterSaleType`       | object   | 售后类型编码对象 |
| `afterSaleTypeName`   | string   | 售后类型名称   |
| `afterSaleStatus`     | object   | 售后状态编码对象 |
| `afterSaleStatusName` | string   | 售后状态名称   |
| `refundAmount`        | number   | 退款金额     |
| `applyReason`         | object   | 售后原因编码对象 |
| `applyReasonName`     | string   | 售后原因名称   |
| `applyDescription`    | string   | 详细说明     |
| `evidenceImages`      | array    | 凭证图片列表   |
| `receiveStatus`       | object   | 收货状态编码对象 |
| `receiveStatusName`   | string   | 收货状态名称   |
| `rejectReason`        | string   | 驳回原因     |
| `adminRemark`         | string   | 管理员备注    |
| `applyTime`           | datetime | 申请时间     |
| `auditTime`           | datetime | 审核时间     |
| `completeTime`        | datetime | 完成时间     |
| `productInfo`         | object   | 商品信息     |
| `timeline`            | array    | 售后时间线    |

#### productInfo

| 字段             | 类型      | 说明    |
|----------------|---------|-------|
| `productId`    | long    | 商品 ID |
| `productName`  | string  | 商品名称  |
| `productImage` | string  | 商品图片  |
| `productPrice` | number  | 商品单价  |
| `quantity`     | integer | 购买数量  |
| `totalPrice`   | number  | 小计金额  |

#### timeline[]

| 字段                 | 类型       | 说明        |
|--------------------|----------|-----------|
| `id`               | long     | 时间线 ID    |
| `eventType`        | object   | 事件类型编码对象  |
| `eventTypeName`    | string   | 事件类型名称    |
| `eventStatus`      | string   | 事件状态      |
| `operatorType`     | object   | 操作方类型编码对象 |
| `operatorTypeName` | string   | 操作方类型名称   |
| `description`      | string   | 事件描述      |
| `createTime`       | datetime | 创建时间      |

---

## 5.2 校验售后资格

### 路径

```text
GET /agent/client/after-sale/eligibility
```

### 查询参数

| 参数名           | 类型     | 必填 | 说明              |
|---------------|--------|----|-----------------|
| `orderNo`     | string | 是  | 订单编号            |
| `orderItemId` | long   | 否  | 订单项 ID，不传表示校验整单 |

### 校验规则

- `orderNo` 不能为空
- `orderItemId` 如果传入，必须 `>= 1`

### 返回 data 结构

| 字段                 | 类型      | 说明                    |
|--------------------|---------|-----------------------|
| `orderNo`          | string  | 订单编号                  |
| `orderItemId`      | long    | 订单项 ID                |
| `scope`            | string  | 校验范围，`ORDER` 或 `ITEM` |
| `orderStatus`      | object  | 当前订单状态编码对象            |
| `orderStatusName`  | string  | 当前订单状态名称              |
| `eligible`         | boolean | 是否满足售后资格              |
| `reasonCode`       | string  | 结果编码                  |
| `reasonMessage`    | string  | 结果说明                  |
| `refundableAmount` | number  | 可退款金额                 |

### scope 取值

| 值       | 含义  |
|---------|-----|
| `ORDER` | 整单  |
| `ITEM`  | 订单项 |

### 当前 reasonCode

| code                        | 含义             |
|-----------------------------|----------------|
| `ELIGIBLE`                  | 满足售后资格         |
| `ORDER_NOT_FOUND`           | 订单不存在或不属于当前用户  |
| `ORDER_ITEM_NOT_FOUND`      | 订单项不存在或不属于当前订单 |
| `ORDER_NOT_PAID`            | 订单未支付          |
| `ORDER_STATUS_INVALID`      | 订单状态异常         |
| `ORDER_STATUS_NOT_ELIGIBLE` | 当前订单状态不允许售后    |
| `AFTER_SALE_IN_PROGRESS`    | 存在进行中的售后       |
| `HISTORY_CONFLICT`          | 历史售后记录冲突       |
| `NO_REFUNDABLE_AMOUNT`      | 无可退款金额         |

## 6. 编码值补充说明

以下为 AI 端较常用的代码值。

### 6.1 订单状态

| code               | 含义  |
|--------------------|-----|
| `PENDING_PAYMENT`  | 待支付 |
| `PENDING_SHIPMENT` | 待发货 |
| `PENDING_RECEIPT`  | 待收货 |
| `COMPLETED`        | 已完成 |
| `REFUNDED`         | 已退款 |
| `AFTER_SALE`       | 售后中 |
| `EXPIRED`          | 已过期 |
| `CANCELLED`        | 已取消 |

### 6.2 支付方式

| code         | 含义         |
|--------------|------------|
| `WALLET`     | 使用钱包余额进行支付 |
| `ALIPAY`     | 使用支付宝进行支付  |
| `WECHAT_PAY` | 使用微信支付进行支付 |
| `BANK_CARD`  | 使用银行卡进行支付  |
| `WAIT_PAY`   | 待支付        |
| `CANCELLED`  | 订单已取消      |

### 6.3 物流状态

| code          | 含义  |
|---------------|-----|
| `NOT_SHIPPED` | 未发货 |
| `IN_TRANSIT`  | 运输中 |
| `DELIVERED`   | 已签收 |
| `EXCEPTION`   | 异常  |

### 6.4 售后类型

| code            | 含义   |
|-----------------|------|
| `REFUND_ONLY`   | 仅退款  |
| `RETURN_REFUND` | 退货退款 |
| `EXCHANGE`      | 换货   |

### 6.5 售后状态

| code         | 含义  |
|--------------|-----|
| `PENDING`    | 待审核 |
| `APPROVED`   | 已通过 |
| `REJECTED`   | 已拒绝 |
| `PROCESSING` | 处理中 |
| `COMPLETED`  | 已完成 |
| `CANCELLED`  | 已取消 |

### 6.6 售后原因

| code               | 含义        |
|--------------------|-----------|
| `ADDRESS_ERROR`    | 收货地址填错了   |
| `NOT_AS_DESCRIBED` | 与描述不符     |
| `INFO_ERROR`       | 信息填错了，重新拍 |
| `DAMAGED`          | 收到商品损坏了   |
| `DELAYED`          | 未按预定时间发货  |
| `OTHER`            | 其它原因      |

### 6.7 收货状态

| code           | 含义   |
|----------------|------|
| `RECEIVED`     | 已收到货 |
| `NOT_RECEIVED` | 未收到货 |

### 6.8 订单事件类型

| code                   | 含义     |
|------------------------|--------|
| `ORDER_CREATED`        | 订单创建   |
| `ORDER_PAID`           | 订单支付   |
| `ORDER_SHIPPED`        | 订单发货   |
| `ORDER_RECEIVED`       | 确认收货   |
| `ORDER_COMPLETED`      | 订单完成   |
| `ORDER_REFUNDED`       | 订单退款   |
| `ORDER_CANCELLED`      | 订单取消   |
| `ORDER_EXPIRED`        | 订单过期   |
| `AFTER_SALE_APPLIED`   | 申请售后   |
| `AFTER_SALE_APPROVED`  | 售后审核通过 |
| `AFTER_SALE_REJECTED`  | 售后审核拒绝 |
| `AFTER_SALE_COMPLETED` | 售后完成   |
| `OTHER`                | 其他     |

### 6.9 操作方类型

| code     | 含义  |
|----------|-----|
| `USER`   | 用户  |
| `ADMIN`  | 管理员 |
| `SYSTEM` | 系统  |

## 7. AI 端对接建议

- 判断调用成功失败时，请以响应体里的 `code` 为准，不要只看 HTTP 状态
- 处理编码字段时，优先读取：
    - `xxxName`
    - 或编码对象里的 `description`
- 商品搜索接口是 `GET` + query string，不是 `POST`
- 商品搜索当前走 ES 检索
- 订单、售后接口全部依赖当前登录用户，不允许自行传 `userId`
- `data.meta` 可忽略，不影响业务使用
- `receiverPhone` 为脱敏值，不适合做精确比对

## 8. 建议的最小调用示例

### 8.1 搜索商品

```text
GET /agent/client/product/search?keyword=感冒灵&pageNum=1&pageSize=10
```

### 8.2 获取订单详情

```text
GET /agent/client/order/O202511130001
```

### 8.3 获取订单物流

```text
GET /agent/client/order/shipping/O202511130001
```

### 8.4 获取订单时间线

```text
GET /agent/client/order/timeline/O202511130001
```

### 8.5 校验是否可取消订单

```text
GET /agent/client/order/cancel-check/O202511130001
```

### 8.6 获取售后详情

```text
GET /agent/client/after-sale/AS202511130001
```

### 8.7 校验整单售后资格

```text
GET /agent/client/after-sale/eligibility?orderNo=O202511130001
```

### 8.8 校验订单项售后资格

```text
GET /agent/client/after-sale/eligibility?orderNo=O202511130001&orderItemId=1001
```
