"""Stage M02 - Thorax Organ Router & PSPNet Segmentation Wrapper."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
from PIL import Image

from src.tier0_errors import OrganRoutingError

SEGMENTATION_SIZE = 512
MIN_THORACIC_PIXELS = 1000
VALID_THORAX_BODY_PARTS = frozenset({"CHEST", "THORAX"})


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2D float32 image to (size, size) using bilinear interpolation."""
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
    """Map arbitrary 2D radiograph intensities into TorchXRayVision [-1024, 1024] range."""
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
    """Wraps TorchXRayVision PSPNet chest anatomy segmenter."""

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
                "M02", "M02_XRV_UNAVAILABLE", f"torchxrayvision is required for PSPNet segmentation: {exc}"
            ) from exc
        model = xrv.baseline_models.chestx_det.PSPNet()
        model = model.to(torch.device(self.device))
        model.eval()
        self._model = model
        return self._model

    def predict_masks(self, xrv_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Return boolean masks keyed by anatomical structure name on a 512x512 grid."""
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
    min_pixel_threshold: int
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "declared_body_part": self.declared_body_part,
            "thoracic_pixels": self.thoracic_pixels,
            "min_pixel_threshold": self.min_pixel_threshold,
            "warnings": list(self.warnings),
        }


class ThoraxRouter:
    """Two-factor Layer 1 gate: declared body part metadata AND segmented thoracic pixels."""

    def __init__(
        self,
        engine: Optional[SegmentationEngine] = None,
        *,
        min_thoracic_pixels: int = MIN_THORACIC_PIXELS,
    ) -> None:
        self.engine = engine or get_default_engine("cpu")
        self.min_thoracic_pixels = int(min_thoracic_pixels)

    def audit(self, image: np.ndarray, declared_body_part: str) -> ThoraxAuditReport:
        norm_bp = (declared_body_part or "").strip().upper()
        if norm_bp not in VALID_THORAX_BODY_PARTS:
            raise OrganRoutingError(
                "M02",
                "M02_BODY_PART_MISMATCH",
                f"Declared body part {declared_body_part!r} is not valid for Thoracic routing.",
                {"declared_body_part": declared_body_part},
            )

        working = to_xrv_range(resize_image(image, SEGMENTATION_SIZE))
        masks = self.engine.predict_masks(working)
        union = np.zeros((SEGMENTATION_SIZE, SEGMENTATION_SIZE), dtype=bool)
        for key in ("Left Lung", "Right Lung", "Heart"):
            if key in masks:
                union |= np.asarray(masks[key], dtype=bool)

        thoracic_pixels = int(union.sum())
        if thoracic_pixels < self.min_thoracic_pixels:
            raise OrganRoutingError(
                "M02",
                "M02_INSUFFICIENT_THORACIC_PIXELS",
                f"Segmented thoracic pixels ({thoracic_pixels}) < threshold ({self.min_thoracic_pixels}).",
                {"thoracic_pixels": thoracic_pixels, "threshold": self.min_thoracic_pixels},
            )

        warnings: List[str] = []
        total_pixels = SEGMENTATION_SIZE * SEGMENTATION_SIZE
        if thoracic_pixels > int(0.95 * total_pixels):
            warnings.append("M02_MASK_NEAR_FULL_GRID_SATURATION")

        return ThoraxAuditReport(
            passed=True,
            declared_body_part=norm_bp,
            thoracic_pixels=thoracic_pixels,
            min_pixel_threshold=self.min_thoracic_pixels,
            warnings=tuple(warnings),
        )
