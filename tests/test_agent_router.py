import pytest
import asyncio
from src.schemas import AgentRequest
from src.agent_router import MasterRouter

@pytest.mark.asyncio
async def test_valid_agent_dispatch():
    router = MasterRouter()
    req = AgentRequest("D01_THORACIC", "A02_LUNG_SEGMENTATION", {"pixels": 120000})
    res = await router.dispatch_single(req)
    assert res.status == "COMPLETED"
    assert res.confidence == 0.95

@pytest.mark.asyncio
async def test_invalid_domain_rejection():
    router = MasterRouter()
    req = AgentRequest("D99_UNKNOWN", "A01_DICOM_PRECHECK", {})
    res = await router.dispatch_single(req)
    assert res.status == "FAILED"
    assert "UNKNOWN_DOMAIN: D99_UNKNOWN" in res.errors

@pytest.mark.asyncio
async def test_concurrent_batch_routing():
    router = MasterRouter()
    reqs = [
        AgentRequest("D01_THORACIC", "A01_DICOM_PRECHECK", {}),
        AgentRequest("D02_CARDIOLOGY", "A05_ECG_PRECHECK", {}),
        AgentRequest("D05_SAFETY_OPS", "A36_AUDIT_LOGGER", {})
    ]
    responses = await router.route_batch(reqs)
    assert len(responses) == 3
    assert all(r.status == "COMPLETED" for r in responses)
