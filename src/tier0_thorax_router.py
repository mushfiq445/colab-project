"""Stage M02 - Thorax Organ Router & PSPNet Segmentation Wrapper."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
import yaml
from PIL import Image

from src.tier0_errors import OrganRoutingError

SEGMENTATION_SIZE = 512
VALID_THORAX_BODY_PARTS = frozenset({"CHEST", "THORAX", "VISUAL_ONLY"})
DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "config" / "organ_thresholds.yaml"


def load_thorax_thresholds(path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
    p = Path(path) if path else DEFAULT_THRESHOLDS_PATH
    if not p.is_file():
        raise OrganRoutingError("M02", "M02_CONFIG_MISSING", f"Threshold config missing: {p}")
    try:
        data = yaml.safe_load(p.read_bytes())
        th = data["thorax"]
        return {
            "min_total_pixels": int(th["min_total_pixels"]),
            "min_left_lung_pixels": int(th["min_left_lung_pixels"]),
            "min_right_lung_pixels": int(th["min_right_lung_pixels"]),
            "max_saturation_pixels": int(th["max_saturation_pixels"]),
        }
    except Exception as exc:
        raise OrganRoutingError("M02", "M02_CONFIG_INVALID", f"Invalid organ_thresholds.yaml: {exc}")


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise OrganRoutingError(
            "M02", "M02_INVALID_IMAGE_SHAPE", f"Expected 2D image array, got {getattr(image, 'shape', None)}."
        )
    if image.shape[0] == size and image.shape[1] == size:
        return image.astype(np.float32, copy=False)
    pil_img = Image.fromarray(image.astype(np.float32), mode="F")
    resized = pil_img.resize((int(size), int(size)), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def to_xrv_range(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise OrganRoutingError("M02", "M02_NON_FINITE_INPUT", "Image contains NaN or Inf values.")
    imin = float(np.min(arr))
    imax = float(np.max(arr))
    if imax <= imin:
        raise OrganRoutingError("M02", "M02_ZERO_VARIANCE_IMAGE", "Image has zero dynamic range.")
    norm01 = (arr - imin) / (imax - imin)
    return (norm01 * 2048.0 - 1024.0).astype(np.float32)


class SegmentationEngine:
    def __init__(self, device: str = "cpu", model: Any = None) -> None:
        self.device = str(device)
        self._model = model

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            import torchxrayvision as xrv
        except ImportError as exc:
            raise OrganRoutingError(
                "M02", "M02_XRV_UNAVAILABLE", f"torchxrayvision is required: {exc}"
            ) from exc
        model = xrv.baseline_models.chestx_det.PSPNet()
        model = model.to(torch.device(self.device))
        model.eval()
        self._model = model
        return self._model

    def predict_masks(self, xrv_image: np.ndarray) -> Dict[str, np.ndarray]:
        import torch
        model = self._ensure_model()
        tensor = torch.from_numpy(np.asarray(xrv_image, dtype=np.float32))[None, None].to(self.device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
        targets = getattr(model, "targets", ["Left Lung", "Right Lung", "Heart"])
        return {str(name): (probs[idx] >= 0.5) for idx, name in enumerate(targets)}


_ENGINE_CACHE: Dict[str, SegmentationEngine] = {}

def get_default_engine(device: str = "cpu") -> SegmentationEngine:
    key = str(device)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = SegmentationEngine(device=key)
    return _ENGINE_CACHE[key]


@dataclass(frozen=True)
class ThoraxAuditReport:
    passed: bool
    declared_body_part: str
    thoracic_pixels: int
    left_lung_pixels: int
    right_lung_pixels: int
    thresholds_used: Dict[str, int]
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    union_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "declared_body_part": self.declared_body_part,
            "thoracic_pixels": self.thoracic_pixels,
            "left_lung_pixels": self.left_lung_pixels,
            "right_lung_pixels": self.right_lung_pixels,
            "thresholds_used": dict(self.thresholds_used),
            "warnings": list(self.warnings),
        }


class ThoraxRouter:
    def __init__(self, engine: Optional[SegmentationEngine] = None) -> None:
        self.engine = engine or get_default_engine("cpu")
        self.thresholds = load_thorax_thresholds()

    def audit(self, image: np.ndarray, declared_body_part: str) -> ThoraxAuditReport:
        norm_bp = (declared_body_part or "").strip().upper()
        if norm_bp not in VALID_THORAX_BODY_PARTS:
            raise OrganRoutingError(
                "M02", "M02_BODY_PART_MISMATCH",
                f"Body part {declared_body_part!r} invalid for Thoracic routing."
            )

        working = to_xrv_range(resize_image(image, SEGMENTATION_SIZE))
        masks = self.engine.predict_masks(working)

        mask_l = np.asarray(masks.get("Left Lung", np.zeros_like(working, dtype=bool)), dtype=bool)
        mask_r = np.asarray(masks.get("Right Lung", np.zeros_like(working, dtype=bool)), dtype=bool)
        mask_h = np.asarray(masks.get("Heart", np.zeros_like(working, dtype=bool)), dtype=bool)
        
        union = mask_l | mask_r | mask_h
        px_l, px_r, px_tot = int(mask_l.sum()), int(mask_r.sum()), int(union.sum())

        if px_tot < self.thresholds["min_total_pixels"]:
            raise OrganRoutingError(
                "M02", "M02_INSUFFICIENT_THORACIC_PIXELS",
                f"Total thoracic pixels {px_tot} < {self.thresholds['min_total_pixels']}"
            )
        if px_l < self.thresholds["min_left_lung_pixels"]:
            raise OrganRoutingError(
                "M02", "M02_INSUFFICIENT_LEFT_LUNG",
                f"Left lung pixels {px_l} < {self.thresholds['min_left_lung_pixels']}"
            )
        if px_r < self.thresholds["min_right_lung_pixels"]:
            raise OrganRoutingError(
                "M02", "M02_INSUFFICIENT_RIGHT_LUNG",
                f"Right lung pixels {px_r} < {self.thresholds['min_right_lung_pixels']}"
            )

        warnings: List[str] = []
        if px_tot > self.thresholds["max_saturation_pixels"]:
            warnings.append("M02_MASK_NEAR_FULL_GRID_SATURATION")

        return ThoraxAuditReport(
            passed=True,
            declared_body_part=norm_bp,
            thoracic_pixels=px_tot,
            left_lung_pixels=px_l,
            right_lung_pixels=px_r,
            thresholds_used=self.thresholds,
            warnings=tuple(warnings),
            union_mask=union,
        )
