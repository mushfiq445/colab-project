"""Tier 0 orchestrator: Pre-Check, Input Validation & Modality Routing.

Public API
----------
``execute_tier0_precheck(input_path, modality_type, device)``

Imaging (``XRAY`` / ``CXR`` / ``DICOM``):  M01 -> M03 -> M02 -> Tier 1 tensor
Biosignal (``ECG`` / ``BIOSIGNAL``):       M04 -> Tier 2 tensor

The function never raises for Tier 0 halting conditions; it returns a structured
dictionary so the caller (and the audit trail) always receives the full diagnostic
context. Unexpected exceptions are captured with their traceback and also reported
as a failed pre-check (fail-closed).

Returned dictionary keys
------------------------
``passed`` (bool), ``modality`` (str), ``route`` (``"TIER1_THORAX"`` | ``"TIER2_BIOSIGNAL"`` | None),
``halted_at`` (stage or None), ``errors`` (list of dicts with stage/code/message/details/trace),
``warnings`` (list[str]), ``stages`` (per-micro-agent result dicts), ``tensor``
(torch.Tensor or None), ``tensor_spec`` (str), ``metadata`` (PHI-free dict).

Tensor contracts
----------------
* Tier 1: ``(1, 1, 224, 224)`` float32 on ``device``, intensity scaled to [-1024, 1024]
  (torchxrayvision convention used by the Phase 1 DenseNet-121 agent).
* Tier 2: ``(1, 12, n_samples)`` float32 on ``device``, band-pass filtered
  (0.5-40 Hz) and z-scored per lead.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from src.tier0_abdomen_router import AbdomenRouter
from src.tier0_biosignal_gate import BiosignalGate, load_ecg, zscore_leads
from src.tier0_dicom_validator import DicomValidator
from src.tier0_errors import InputError, Tier0Error
from src.tier0_thorax_router import (
    SegmentationEngine,
    ThoraxRouter,
    get_default_engine,
    resize_image,
    to_xrv_range,
)

LOGGER = logging.getLogger(__name__)

IMAGING_MODALITIES = frozenset({"XRAY", "CXR", "DICOM", "DX", "CR"})
BIOSIGNAL_MODALITIES = frozenset({"ECG", "BIOSIGNAL", "12LEAD_ECG"})
TIER1_INPUT_SIZE = 224


def _new_result(modality: str) -> Dict[str, Any]:
    return {
        "passed": False, "modality": modality, "route": None, "halted_at": None,
        "errors": [], "warnings": [], "stages": {}, "tensor": None, "tensor_spec": "",
        "metadata": {},
    }


def _record_error(result: Dict[str, Any], stage: str, exc: BaseException) -> None:
    entry: Dict[str, Any]
    if isinstance(exc, Tier0Error):
        entry = exc.to_dict()
        stage = exc.stage
    else:
        entry = {"stage": stage, "code": "UNEXPECTED_EXCEPTION", "message": repr(exc), "details": {}}
    entry["trace"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    result["errors"].append(entry)
    result["halted_at"] = stage
    result["passed"] = False
    result["tensor"] = None
    LOGGER.error("Tier0 halt at %s: %s", stage, entry["message"])


def _run_imaging(
    result: Dict[str, Any],
    path: Path,
    device: torch.device,
    engine: Optional[SegmentationEngine],
) -> None:
    stage = "M01"
    try:
        m01 = DicomValidator().validate(path)
        result["stages"]["M01"] = {"passed": True, **m01.report.to_dict()}
        result["warnings"].extend(m01.report.warnings)
        result["metadata"] = {
            "modality": m01.report.modality, "body_part_examined": m01.report.body_part_examined,
            "view_position": m01.report.view_position, "rows": m01.report.rows,
            "columns": m01.report.columns, "bits_stored": m01.report.bits_stored,
            "hipaa_safe_harbor_compliant": m01.report.hipaa_safe_harbor_compliant,
        }
        image = m01.pixel_array

        stage = "M03"
        m03 = AbdomenRouter().enforce(image)
        result["stages"]["M03"] = {"passed": True, **m03.to_dict()}

        stage = "M02"
        seg = engine if engine is not None else get_default_engine(str(device))
        m02 = ThoraxRouter(seg).audit(image, m01.report.body_part_examined)
        result["stages"]["M02"] = m02.to_dict()
        result["warnings"].extend(m02.warnings)

        stage = "TENSOR"
        arr = to_xrv_range(resize_image(image, TIER1_INPUT_SIZE))
        result["tensor"] = torch.from_numpy(arr)[None, None].to(device=device, dtype=torch.float32)
        result["tensor_spec"] = f"(1,1,{TIER1_INPUT_SIZE},{TIER1_INPUT_SIZE}) float32, range [-1024,1024]"
        result["route"] = "TIER1_THORAX"
        result["passed"] = True
    except Exception as exc:  # fail-closed
        if stage in ("M01", "M02", "M03"):
            result["stages"].setdefault(stage, {"passed": False})
            result["stages"][stage]["passed"] = False
        _record_error(result, stage, exc)


def _run_biosignal(result: Dict[str, Any], path: Path, device: torch.device) -> None:
    stage = "M04"
    try:
        ecg = load_ecg(path)
        m04 = BiosignalGate().enforce(ecg)
        result["stages"]["M04"] = {"passed": True, **m04.to_dict()}
        assert m04.filtered_mv is not None  # guaranteed when passed
        result["tensor"] = torch.from_numpy(zscore_leads(m04.filtered_mv))[None].to(device=device, dtype=torch.float32)
        result["tensor_spec"] = f"(1,12,{m04.n_samples}) float32, band-pass 0.5-40 Hz, per-lead z-score"
        result["metadata"] = {"sampling_rate_hz": m04.sampling_rate_hz, "n_samples": m04.n_samples,
                              "pooled_snr_db": m04.snr_db}
        result["route"] = "TIER2_BIOSIGNAL"
        result["passed"] = True
    except Exception as exc:
        result["stages"].setdefault("M04", {"passed": False})
        result["stages"]["M04"]["passed"] = False
        if isinstance(exc, Tier0Error) and "leads" in exc.details:
            result["stages"]["M04"].update(exc.details)
        _record_error(result, stage, exc)


def execute_tier0_precheck(
    input_path: Union[str, Path],
    modality_type: str,
    device: Union[str, torch.device] = "cpu",
    *,
    thorax_engine: Optional[SegmentationEngine] = None,
) -> Dict[str, Any]:
    """Run the Tier 0 pre-check and return a structured result (see module docstring).

    ``thorax_engine`` optionally injects the segmentation backend (defaults to the cached
    torchxrayvision PSPNet engine used by Phase 1).
    """
    modality = str(modality_type).strip().upper()
    result = _new_result(modality)
    try:
        dev = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        _record_error(result, "ORCHESTRATOR", InputError("ORCHESTRATOR", "INVALID_DEVICE", f"Invalid device {device!r}: {exc}"))
        return result

    path = Path(input_path)
    if modality in IMAGING_MODALITIES:
        _run_imaging(result, path, dev, thorax_engine)
    elif modality in BIOSIGNAL_MODALITIES:
        _run_biosignal(result, path, dev)
    else:
        _record_error(
            result, "ORCHESTRATOR",
            InputError("ORCHESTRATOR", "INVALID_MODALITY_TYPE",
                       f"Unknown modality_type {modality_type!r}; expected one of "
                       f"{sorted(IMAGING_MODALITIES | BIOSIGNAL_MODALITIES)}."),
        )
    return result

