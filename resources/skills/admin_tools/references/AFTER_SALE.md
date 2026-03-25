# After Sale Tools Reference

## 领域说明

售后领域工具用于查询售后申请列表和售后详情。  
返回结构以 Python 工具实际返回的 `data` 为准，不包含 Java 原始响应外层的 `code/message/timestamp`。

## 工具清单

| 工具名                 | 适用场景                   | 关键参数                                                                                             |
|---------------------|------------------------|--------------------------------------------------------------------------------------------------|
| `after_sale_list`   | 查询待处理售后、定位某订单或某用户的售后范围 | `page_num`、`page_size`、`after_sale_type`、`after_sale_status`、`order_no`、`user_id`、`apply_reason` |
| `after_sale_detail` | 查看单个售后申请完整详情           | `after_sale_id: int`                                                                             |

## after_sale_list

### 实际返回结构

分页对象：

- `total`
- `pageNum`
- `pageSize`
- `rows`

`rows[]` 中每项通常包含：

- `id`
- `afterSaleNo`
- `orderNo`
- `userId`
- `userNickname`
- `productName`
- `productImage`
- `afterSaleType`
- `afterSaleStatus`
- `refundAmount`
- `applyReason`
- `applyTime`
- `auditTime`

### 关键字段解释

- `afterSaleType`、`afterSaleStatus`、`applyReason` 是业务编码字段。
- 列表页适合快速定位售后单，不包含完整时间线和管理员备注。

### 常见可回答问题

- 当前有哪些待审核或处理中售后
- 某个订单是否有售后申请
- 某个用户近期有哪些售后单

## after_sale_detail

### 实际返回结构

返回单个售后详情对象，常见字段：

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

`productInfo` 常见字段：

- `productId`
- `productName`
- `productImage`
- `productPrice`
- `quantity`
- `totalPrice`

`timeline[]` 中每项常见字段：

- `id`
- `eventType`
- `eventTypeName`
- `eventStatus`
- `operatorType`
- `operatorTypeName`
- `description`
- `createTime`

### 关键字段解释

- `afterSaleTypeName`、`afterSaleStatusName`、`applyReasonName`、`receiveStatusName` 是更适合直接面向用户说明的中文名称。
- `evidenceImages` 是凭证图片数组。
- `rejectReason`、`adminRemark` 适合说明后台处理结论和原因。
- `timeline` 适合整理成售后处理流程。

### 常见可回答问题

- 某个售后单当前是什么状态
- 用户为什么申请售后、提交了哪些凭证
- 后台为什么拒绝或通过该售后
- 这个售后处理到哪一步了
