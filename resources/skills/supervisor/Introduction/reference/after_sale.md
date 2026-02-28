# 售后域 (After-sale) 专家指南

## 领域概述

`after_sale_tool_agent` 负责商城售后申请全流程监控。它是典型的“中继节点”，连接订单、用户与商品三大核心域。

---

## 核心能力与返回界限

### 1. 售后多维检索
- **用途**：按类型、状态、原因（如损坏、过期）筛选售后。
- **返回**：售后概览（ID、类型、状态、金额）。

### 2. 售后详情 (get_admin_after_sale_detail)
- **用途**：获取单笔售后的最完整视图，包括处理历史和凭证。
- **返回的关键 ID (用于全链路补查)**：
  - `orderId` & `orderNo`：关联的原始订单 -> 转至 **Order Agent**。
  - `userId`：申请售后的用户 -> 转至 **User Agent**。
  - `productId`：售后涉及的具体商品 -> 转至 **Product Agent**。

**⚠️ 数据边界（必读）**：
> 售后详情中的数据是 **“申请快照”**。
> **它不持有以下数据**：
> - 订单的完整物流轨迹。
> - 用户的钱包当前余额。
> - 商品的实时库存或最新价格。
> **处理建议**：如需深入分析售后背后的原因，必须利用 ID 进行跨域调度。

### 3. 处理历史与凭证
- **用途**：查看审核节点、拒绝原因及用户上传的照片凭证。

---

## 典型编排场景

### 场景：全面排查一笔售后
1. **Step 1 (After-sale)**: `after_sale_tool_agent("查询售后单 5001 的深度详情，提取 orderId, userId, productId")`。
2. **Step 2 (Parallel)**:
   - 调用 `order_tool_agent` 查物流。
   - 调用 `user_tool_agent` 查信用与钱包。
   - 调用 `product_tool_agent` 查规格。

---

## 输入规范 (Supervisor 必须遵守)

- **单 ID 策略**：详情查询必须使用具体的 `after_sale_id` 数值。
- **状态敏感**：重点关注 `REJECTED` 或 `PENDING` 状态下的原因字段。

---

## 跨域 ID 映射表

| 用户后续提问 | 依赖 ID | 下一跳 Agent |
| :--- | :--- | :--- |
| “这笔订单什么时候发的货？” | `orderId` | `order_tool_agent` (查发货记录) |
| “退款退到用户钱包了吗？” | `userId` | `user_tool_agent` (查流水) |
| “退货的商品是什么规格？” | `productId` | `product_tool_agent` |
