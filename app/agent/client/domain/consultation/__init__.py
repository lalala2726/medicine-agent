"""Client 病情咨询子图包。"""

from app.agent.client.domain.consultation.agent import consultation_agent
from app.agent.client.domain.consultation.state import ConsultationState

__all__ = [
    "consultation_agent",
    "ConsultationState",
]
