"""Task 7: Micro-Agent Data Schemas."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class AgentRequest:
    domain_id: str         # e.g., 'D01_THORACIC', 'D02_CARDIOLOGY'
    agent_id: str          # e.g., 'A01_LUNG_SEGMENTATION'
    payload: Dict[str, Any]
    priority: int = 1      # 1 (Critical) to 5 (Background)

@dataclass
class AgentResponse:
    agent_id: str
    status: str            # 'COMPLETED', 'FAILED', 'DEFERRED'
    confidence: float
    data: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
