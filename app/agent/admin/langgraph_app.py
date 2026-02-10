"""
LangGraph CLI entrypoint.

Expose a module-level compiled graph so `langgraph dev` can load it from
`langgraph.json` via `./app/agent/admin/langgraph_app.py:graph`.
"""

from app.agent.admin.workflow import build_graph

# LangGraph CLI expects a top-level graph object.
graph = build_graph()
