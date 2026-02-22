from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from agent.assistant.model_switch import model_switch
from app.agent.assistant.state import AgentState, ExecutionTraceState
from app.agent.assistant.tools.analytics_tool import analytics_tool_agent
from app.agent.assistant.tools.chart_tool import chart_tool_agent
from app.agent.assistant.tools.order_tool import order_tool_agent
from app.agent.assistant.tools.product_tool import product_tool_agent
from app.core.langsmith import traceable
from app.core.llm import create_chat_model
from app.schemas.prompt import base_prompt
from app.services.token_usage_service import append_trace_and_refresh_token_usage
from app.utils.streaming_utils import invoke_with_trace

_SUPERVISOR_PROMPT = """
    你是药品商城后台管理助手的 supervisor 节点。
    你的职责是根据用户意图决策是否调用子工具并输出最终结果。

    ## 工具调用策略
    | 场景 | 调用工具 | 说明 |
    |-----|---------|-----|
    | 订单查询/分析 | order_tool_agent | 订单状态、详情、统计等 |
    | 商品查询/管理 | product_tool_agent | 商品信息、库存、分类等 |
    | 运营数据分析 | analytics_tool_agent | 销售报表、趋势分析等 |
    | 图表生成需求 | chart_tool_agent | 获取图表模板和字段规范 |
    | 非业务闲聊 | 直接回答 | 无需调用工具 |

    ## 图表生成规范（严格遵守）

    ### 支持的图表类型（18种）
    line, area, column, bar, pie, histogram, scatter, wordcloud, treemap,
    dualaxes, radar, funnel, mindmap, networkgraph, flowdiagram,
    organizationchart, indentedtree, fishbonediagram

    ### 输出格式要求
    1. **代码块标识**：必须使用精确的图表类型作为语言标识
       ```
       正确示例：```line、```pie、```fishbonediagram
       错误示例：```chart、```json、```图表
       ```

    2. **JSON结构**：代码块内容必须是合法JSON，字段严格遵循模板规范
       - 必填字段不能省略
       - 数值字段使用数字类型（禁止包含单位或运算符号如 %、+、k）
       - 文本字段使用字符串类型

    3. **生成流程**：
       - 需要生成图表时，必须先调用 chart_tool_agent 获取模板
       - 根据业务数据填充模板，保持字段结构不变
       - 一个回复中可包含多个图表代码块

    ### 各图表关键字段速查
    | 图表 | 必填数据字段 | 说明 |
    |-----|-------------|-----|
    | line | data[{time, value}] | 时间序列数据 |
    | area | data[{time, value}] | 同line，支持stack堆叠 |
    | column/bar | data[{category, value}] | 分类比较数据 |
    | pie | data[{category, value}] | 占比数据（value不用百分比） |
    | histogram | data[数值数组] | 原始数据，自动分箱 |
    | scatter | data[{x, y}] | 双变量关系数据 |
    | wordcloud | data[{text, value}] | 词频数据 |
    | treemap | data[{name, value, children?}] | 层级占比数据 |
    | dualaxes | categories[] + series[] | 双Y轴组合图 |
    | radar | data[{name, value}] | 多维评价数据 |
    | funnel | data[{category, value}] | 转化漏斗数据 |
    | mindmap | data{name, children[]} | 思维导图结构 |
    | networkgraph | data{nodes[], edges[]} | 节点关系图 |
    | flowdiagram | data{nodes[], edges[]} | 流程步骤图 |
    | organizationchart | data{name, children[]} | 组织架构树 |
    | indentedtree | data{name, children[]} | 缩进树形结构 |
    | fishbonediagram | data{name, children[]} | 鱼骨因果分析 |

    ## 强约束
    1. 严禁编造工具未返回的数据。
    2. 优先调用工具获取真实数据，再生成最终答复。
    3. 图表字段必须严格遵循模板规范，禁止添加自定义字段。
    4. 输出简洁清晰，不暴露内部调度细节。
""" + base_prompt


@traceable(name="Supervisor Agent Node", run_type="chain")
def supervisor_agent(state: AgentState) -> dict[str, Any]:
    model_name = model_switch(state)

    llm = create_chat_model(
        model=model_name,
        temperature=1.3,
    )
    history_messages = list(state.get("history_messages") or [])

    messages = [SystemMessage(content=_SUPERVISOR_PROMPT), *history_messages]
    trace = invoke_with_trace(
        llm,
        messages,
        tools=[order_tool_agent, product_tool_agent, analytics_tool_agent, chart_tool_agent],
    )
    text = str(trace.get("text") or "").strip()
    trace_item = ExecutionTraceState(
        node_name="supervisor_agent",
        model_name=model_name,
        output_text=text,
        llm_used=True,
        llm_usage_complete=bool(trace.get("is_usage_complete", False)),
        llm_token_usage=trace.get("usage"),
        tool_calls=list(trace.get("tool_calls") or []),
    )
    execution_traces, token_usage = append_trace_and_refresh_token_usage(
        state.get("execution_traces"),
        trace_item,
    )
    return {
        "result": text,
        "messages": [AIMessage(content=text)],
        "execution_traces": execution_traces,
        "token_usage": token_usage,
    }
