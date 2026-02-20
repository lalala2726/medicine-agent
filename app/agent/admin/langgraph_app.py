"""
LangGraph CLI entrypoint for supervisor architecture.
"""

from app.agent.admin.workflow import build_graph

graph = build_graph()

