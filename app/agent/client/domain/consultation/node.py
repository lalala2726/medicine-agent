"""Client consultation 兼容转发模块。"""

from app.agent.client.domain.consultation.agent import consultation_agent
from app.agent.client.domain.consultation.graph import (
    _CONSULTATION_GRAPH,
    _route_after_parallel_merge,
    _route_from_status,
    build_consultation_graph,
    consultation_collecting_fanout_node,
    consultation_collecting_response_node,
    consultation_parallel_merge_node,
    consultation_stream_response_node,
)
from app.agent.client.domain.consultation.nodes.comfort_node import consultation_comfort_node
from app.agent.client.domain.consultation.nodes.final_diagnosis_node import consultation_final_diagnosis_node
from app.agent.client.domain.consultation.nodes.question_node import consultation_question_card_node
from app.agent.client.domain.consultation.nodes.status_node import consultation_status_node

__all__ = [
    "_CONSULTATION_GRAPH",
    "_route_after_parallel_merge",
    "_route_from_status",
    "build_consultation_graph",
    "consultation_agent",
    "consultation_collecting_fanout_node",
    "consultation_collecting_response_node",
    "consultation_comfort_node",
    "consultation_final_diagnosis_node",
    "consultation_parallel_merge_node",
    "consultation_question_card_node",
    "consultation_status_node",
    "consultation_stream_response_node",
]
