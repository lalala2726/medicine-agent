from langgraph.constants import START
from langgraph.graph import StateGraph

from app.agent.admin.agent_state import AgentState
from app.agent.admin.chart_agent import chart_agent
from app.agent.admin.excel_agent import excel_agent
from app.agent.admin.order_agent import order_agent
from app.agent.admin.supervisor_agent import supervisor_agent


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("chart_agent", chart_agent)
    graph.add_node("excel_agent", excel_agent)
    graph.add_node("order_agent", order_agent)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("router", router)
    graph.add_edge(START, "supervisor_agent")
    graph.add_edge("supervisor_agent", "router")

    return graph.compile()


def router(state: AgentState):
    pass
