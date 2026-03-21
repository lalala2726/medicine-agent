"""Client consultation 真实节点实现集合。"""

from app.agent.client.domain.consultation.nodes.comfort_node import consultation_comfort_node
from app.agent.client.domain.consultation.nodes.final_diagnosis_node import consultation_final_diagnosis_node
from app.agent.client.domain.consultation.nodes.question_node import consultation_question_card_node
from app.agent.client.domain.consultation.nodes.status_node import consultation_status_node
from app.agent.client.domain.consultation.nodes.stream_response_node import consultation_stream_response_node

__all__ = [
    "consultation_comfort_node",
    "consultation_final_diagnosis_node",
    "consultation_question_card_node",
    "consultation_status_node",
    "consultation_stream_response_node",
]
