# After Sale References

## open_user_after_sale_list

适用场景：

- 用户明确要求打开、进入、查看售后列表
- 用户希望打开待审核、处理中、已完成等特定状态的售后列表

关键参数：

- `afterSaleStatus`
    - 可选
    - 允许值：`PENDING`、`APPROVED`、`REJECTED`、`PROCESSING`、`COMPLETED`、`CANCELLED`
    - 用户未指定状态时不要传

返回值：

- 返回一段给用户看的确认文案，例如“已为你打开售后列表”“已为你打开处理中售后列表”
- 同时向前端下发页面跳转动作：
    - `target = user_after_sale_list`
    - `payload = {"afterSaleStatus": ...}`

常见可回答问题：

- “打开我的售后”
- “帮我看处理中的售后”
- “我想进待审核售后列表”

## get_after_sale_detail

适用场景：

- 用户已经提供售后单号，想查状态、退款金额、驳回原因、处理时间线、凭证图片

关键参数：

- `after_sale_no`：必填，售后单号

实际返回结构：

- 顶层字段：
    - `id`
    - `afterSaleNo`
    - `orderId`
    - `orderNo`
    - `orderItemId`
    - `userId`
    - `userNickname`
    - `afterSaleType`
    - `afterSaleTypeName`
    - `afterSaleStatus`
    - `afterSaleStatusName`
    - `refundAmount`
    - `applyReason`
    - `applyReasonName`
    - `applyDescription`
    - `evidenceImages`
    - `receiveStatus`
    - `receiveStatusName`
    - `rejectReason`
    - `adminRemark`
    - `applyTime`
    - `auditTime`
    - `completeTime`
    - `productInfo`
    - `timeline`
- `productInfo`：
    - `productId`
    - `productName`
    - `productImage`
    - `productPrice`
    - `quantity`
    - `totalPrice`
- `timeline[]`：
    - `id`
    - `eventType`
    - `eventTypeName`
    - `eventStatus`
    - `operatorType`
    - `operatorTypeName`
    - `description`
    - `createTime`

关键解释：

- `afterSaleType` / `afterSaleTypeName` 表示仅退款、退货退款、换货等类型
- `receiveStatus` / `receiveStatusName` 表示用户收货状态
- `rejectReason` 表示审核拒绝原因

常见可回答问题：

- “售后单 ASxxxx 现在到哪一步了”
- “退款金额是多少”
- “为什么被驳回”
- “这次售后处理过哪些节点”

## check_after_sale_eligibility

适用场景：

- 用户想确认某笔订单或某个订单项还能不能申请退款、退货或换货

关键参数：

- `order_no`：必填，订单编号
- `order_item_id`：可选，订单项 ID；只针对单个商品项判断时传

实际返回结构：

- `orderNo`
- `orderItemId`
- `scope`
- `orderStatus`
- `orderStatusName`
- `eligible`
- `reasonCode`
- `reasonMessage`
- `refundableAmount`

关键解释：

- `scope`：`ORDER` 表示整单校验，`ITEM` 表示订单项校验
- `eligible`：是否满足售后资格
- `refundableAmount`：当前可退款金额

常见可回答问题：

- “这单还能退吗”
- “这个商品还能换货吗”
- “最多能退多少钱”

继续追问规则：

- 没有订单号时，不要直接做资格校验，先要订单号
- 用户在问售后进度时，如果已有售后单号，优先查 `get_after_sale_detail`
