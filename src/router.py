"""Tier 0 orchestrator: Pre-Check, Input Validation & Modality Routing."""
from __future__ import annotations
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from src.tier0_abdomen_router import AbdomenRouter
from src.tier0_biosignal_gate import BiosignalGate, load_ecg, zscore_leads
from src.tier0_dicom_validator import DicomValidator
from src.tier0_errors import InputError, Tier0Error
from src.tier0_thorax_router import (
    SEGMENTATION_SIZE, SegmentationEngine, ThoraxRouter,
    get_default_engine, resize_image, to_xrv_range, load_thorax_thresholds
)

LOGGER = logging.getLogger(__name__)

IMAGING_MODALITIES = frozenset({"XRAY", "CXR", "DICOM", "DX", "CR"})
BIOSIGNAL_MODALITIES = frozenset({"ECG", "BIOSIGNAL", "12LEAD_ECG"})
TIER1_INPUT_SIZE = 224

def _new_result(modality: str) -> Dict[str, Any]:
    return {
        "passed": False, "modality": modality, "route": None,
        "halted_at": None, "errors": [], "warnings": [],
        "stages": {}, "tensor": None, "tensor_spec": "",
        "metadata": {}, "masks": {}
    }

def _record_error(result: Dict[str, Any], stage: str, exc: BaseException) -> None:
    entry = exc.to_dict() if isinstance(exc, Tier0Error) else {
        "stage": stage, "code": "UNEXPECTED_EXCEPTION", "message": repr(exc), "details": {}
    }
    entry["trace"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    result["errors"].append(entry)
    result["halted_at"] = stage
    result["passed"] = False
    result["tensor"] = None
    LOGGER.error("Tier0 halt at %s: %s", stage, entry["message"])

def _run_imaging(result: Dict[str, Any], path: Path, device: torch.device, engine: Optional[SegmentationEngine]) -> None:
    stage = "M01"
    suffix = path.suffix.lower()
    try:
        if suffix in (".jpg", ".jpeg", ".png"):
            raw_pil = Image.open(path).convert("L")
            image = np.array(raw_pil, dtype=np.float32)
            result["stages"]["M01"] = {
                "passed": True, "file_name": path.name, "modality": "DX",
                "body_part_examined": "VISUAL_ONLY", "bits_stored": 8,
                "hipaa_safe_harbor_compliant": True,
                "warnings": ["STANDARD_IMAGE_INGESTED_NO_DICOM_HEADER"],
            }
            body_part = "VISUAL_ONLY"
        else:
            m01 = DicomValidator().validate(path)
            result["stages"]["M01"] = {"passed": True, **m01.report.to_dict()}
            result["warnings"].extend(m01.report.warnings)
            result["metadata"].update(m01.report.to_dict())
            image = m01.pixel_array
            body_part = m01.report.body_part_examined

        stage = "M03"
        m03 = AbdomenRouter().enforce(image)
        result["stages"]["M03"] = {"passed": True, **m03.to_dict()}

        stage = "M02"
        m02 = ThoraxRouter(engine).audit(image, body_part)
        result["stages"]["M02"] = m02.to_dict()
        result["warnings"].extend(m02.warnings)
        result["masks"]["thorax_union"] = m02.union_mask

        stage = "TENSOR"
        arr = to_xrv_range(resize_image(image, TIER1_INPUT_SIZE))
        result["tensor"] = torch.from_numpy(arr)[None, None].to(device=device, dtype=torch.float32)
        result["tensor_spec"] = f"(1,1,{TIER1_INPUT_SIZE},{TIER1_INPUT_SIZE}) float32"
        result["route"] = "TIER1_THORAX"
        result["passed"] = True
    except Exception as exc:
        if stage in ("M01", "M02", "M03"):
            result["stages"].setdefault(stage, {"passed": False})
            result["stages"][stage]["passed"] = False
        _record_error(result, stage, exc)

def _run_biosignal(result: Dict[str, Any], path: Path, device: torch.device) -> None:
    stage = "M04"
    try:
        m04 = BiosignalGate().enforce(load_ecg(path))
        result["stages"]["M04"] = {"passed": True, **m04.to_dict()}
        result["tensor"] = torch.from_numpy(zscore_leads(m04.filtered_mv))[None].to(device=device, dtype=torch.float32)
        result["metadata"].update(m04.to_dict())
        result["route"] = "TIER2_BIOSIGNAL"
        result["passed"] = True
    except Exception as exc:
        result["stages"].setdefault("M04", {"passed": False})
        result["stages"][stage]["passed"] = False
        _record_error(result, stage, exc)

def execute_tier0_precheck(
    input_path: Union[str, Path], modality_type: str, device: Union[str, torch.device] = "cpu",
    *, thorax_engine: Optional[SegmentationEngine] = None,
) -> Dict[str, Any]:
    modality = str(modality_type).strip().upper()
    result = _new_result(modality)
    try:
        dev = torch.device(device)
    except Exception as exc:
        _record_error(result, "ORCHESTRATOR", InputError("ORCHESTRATOR", "INVALID_DEVICE", str(exc)))
        return result

    path = Path(input_path)
    if modality in IMAGING_MODALITIES:
        _run_imaging(result, path, dev, thorax_engine)
    elif modality in BIOSIGNAL_MODALITIES:
        _run_biosignal(result, path, dev)
    else:
        _record_error(result, "ORCHESTRATOR", InputError("ORCHESTRATOR", "INVALID_MODALITY_TYPE", "Unsupported."))
    return result

# Compatibility wrappers for legacy tests
def load_pspnet_segmenter(device: Union[str, torch.device]): return get_default_engine(str(device))
def validate_two_factor_organ(declared_body_part: str, thoracic_pixels: int, min_pixel_threshold: int = 100000) -> bool:
    norm = (declared_body_part or "").strip().upper()
    return (norm in ("CHEST", "THORAX", "VISUAL_ONLY")) and (thoracic_pixels >= min_pixel_threshold)
def segment_thoracic_anatomy(img_path: str, segmenter: Any, device: Any):
    raw = Image.open(img_path).convert("L")
    masks = segmenter.predict_masks(to_xrv_range(resize_image(np.array(raw, dtype=np.float32), SEGMENTATION_SIZE)))
    union = np.zeros((SEGMENTATION_SIZE, SEGMENTATION_SIZE), dtype=bool)
    for k in ("Left Lung", "Right Lung", "Heart"):
        if k in masks: union |= masks[k]
    return union, int(union.sum())
