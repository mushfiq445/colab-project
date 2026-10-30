"""M02 - Thorax Visual Router & Two-Factor Consensus (Tier 0).

Cross-audits the DICOM ``BodyPartExamined`` header (factor 1) against a visual
PSPNet segmentation of the lungs/heart (factor 2, torchxrayvision ChestX-Det baseline).

Decision table
--------------
==========================  ====================  =====================================
Visual (>= min_pixels)      Metadata              Outcome
==========================  ====================  =====================================
fail (< min_pixels)         any                   HARD STOP (ThoraxHardStopError)
pass                        CHEST / THORAX        PASS, consensus ``CONFIRMED``
pass                        absent                PASS ``VISUAL_ONLY`` (warning) if
                                                  ``allow_missing_metadata`` else halt
pass                        other body part       HALT (ConsensusConflictError)
==========================  ====================  =====================================

The pixel-mass threshold is evaluated on the 512 x 512 working grid of the PSPNet
engine (``SEGMENTATION_SIZE``), which is the resolution the engine was trained at.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, Tuple

import numpy as np
from scipy import ndimage

from src.tier0_errors import ConsensusConflictError, InputError, ThoraxHardStopError

LOGGER = logging.getLogger(__name__)
STAGE = "M02"

SEGMENTATION_SIZE = 512
XRV_INTENSITY_RANGE = (-1024.0, 1024.0)
MIN_THORACIC_PIXELS = 1_000
THORAX_BODY_PARTS: FrozenSet[str] = frozenset({"CHEST", "THORAX"})
THORACIC_MASK_KEYS: Tuple[str, ...] = ("Left Lung", "Right Lung", "Heart")


def to_xrv_range(image: np.ndarray) -> np.ndarray:
    """Min-max scale a 2-D image to the torchxrayvision range [-1024, 1024] (float32)."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise InputError(STAGE, "BAD_IMAGE_RANK", f"Expected a 2-D image, got shape {img.shape}.")
    if not np.isfinite(img).all():
        raise InputError(STAGE, "NON_FINITE_PIXELS", "Image contains NaN/Inf.")
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        raise InputError(STAGE, "CONSTANT_IMAGE", "Image has zero dynamic range.")
    unit = (img - lo) / (hi - lo)
    low, high = XRV_INTENSITY_RANGE
    return (unit * (high - low) + low).astype(np.float32)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize of a 2-D image to ``size`` x ``size`` (float32)."""
    h, w = image.shape
    if (h, w) == (size, size):
        return image.astype(np.float32, copy=False)
    out = ndimage.zoom(image.astype(np.float32), (size / h, size / w), order=1, mode="nearest")
    if out.shape != (size, size):  # guard against rounding in zoom
        out = out[:size, :size]
        pad = ((0, size - out.shape[0]), (0, size - out.shape[1]))
        out = np.pad(out, pad, mode="edge")
    return out.astype(np.float32)


class SegmentationEngine(Protocol):
    """Anatomical segmentation backend.

    ``predict_masks`` receives a float32 (512, 512) array already scaled to
    [-1024, 1024] and returns boolean masks (512, 512) keyed by structure name.
    """

    def predict_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]: ...


class XRVPSPNetEngine:
    """Production engine: torchxrayvision ChestX-Det PSPNet baseline."""

    def __init__(self, device: str = "cpu", threshold: float = 0.5) -> None:
        import torch
        import torchxrayvision as xrv

        self._torch = torch
        self._device = torch.device(device)
        self._threshold = threshold
        self._model = xrv.baseline_models.chestx_det.PSPNet().to(self._device).eval()
        self._targets: List[str] = list(self._model.targets)

    @property
    def targets(self) -> List[str]:
        return list(self._targets)

    def predict_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        torch = self._torch
        if image.shape != (SEGMENTATION_SIZE, SEGMENTATION_SIZE):
            raise InputError(STAGE, "BAD_SEGMENTATION_INPUT", f"PSPNet expects 512x512 input, got {image.shape}.")
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(self._device)
        with torch.inference_mode():
            out = self._model(tensor)
            if float(out.min()) < 0.0 or float(out.max()) > 1.0:  # logits -> probabilities
                out = torch.sigmoid(out)
            masks = (out[0] >= self._threshold).cpu().numpy()
        if masks.shape[0] != len(self._targets):
            raise InputError(
                STAGE, "ENGINE_OUTPUT_MISMATCH",
                f"PSPNet returned {masks.shape[0]} channels for {len(self._targets)} targets.",
            )
        return {name: masks[i] for i, name in enumerate(self._targets)}


@lru_cache(maxsize=4)
def get_default_engine(device: str = "cpu") -> XRVPSPNetEngine:
    """Process-wide cached PSPNet engine (weights load once per device)."""
    return XRVPSPNetEngine(device=device)


@dataclass(frozen=True)
class ThoraxRoutingResult:
    passed: bool
    consensus: str  # CONFIRMED | VISUAL_ONLY
    thoracic_pixels: int
    left_lung_pixels: int
    right_lung_pixels: int
    heart_pixels: int
    min_pixels: int
    metadata_body_part: Optional[str]
    metadata_factor: str  # MATCH | MISSING | MISMATCH
    visual_factor: str  # PASS | FAIL
    warnings: List[str] = field(default_factory=list)
    thoracic_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed, "consensus": self.consensus,
            "thoracic_pixels": self.thoracic_pixels, "left_lung_pixels": self.left_lung_pixels,
            "right_lung_pixels": self.right_lung_pixels, "heart_pixels": self.heart_pixels,
            "min_pixels": self.min_pixels, "metadata_body_part": self.metadata_body_part,
            "metadata_factor": self.metadata_factor, "visual_factor": self.visual_factor,
            "warnings": list(self.warnings),
        }


class ThoraxRouter:
    """M02 two-factor consensus router."""

    def __init__(
        self,
        engine: SegmentationEngine,
        min_pixels: int = MIN_THORACIC_PIXELS,
        allow_missing_metadata: bool = True,
        accepted_body_parts: FrozenSet[str] = THORAX_BODY_PARTS,
        mask_keys: Tuple[str, ...] = THORACIC_MASK_KEYS,
    ) -> None:
        self.engine = engine
        self.min_pixels = int(min_pixels)
        self.allow_missing_metadata = allow_missing_metadata
        self.accepted_body_parts = frozenset(p.upper() for p in accepted_body_parts)
        self.mask_keys = tuple(mask_keys)

    def audit(self, image: np.ndarray, body_part_examined: Optional[str]) -> ThoraxRoutingResult:
        """Audit a full-resolution 2-D image (MONOCHROME2 semantics) plus its header body part."""
        working = to_xrv_range(resize_image(np.asarray(image, dtype=np.float32), SEGMENTATION_SIZE))
        masks = self.engine.predict_masks(working)

        missing = [k for k in self.mask_keys if k not in masks]
        if missing:
            raise InputError(STAGE, "SEGMENTATION_KEYS_MISSING", f"Engine output lacks required structures: {missing}")
        for key in self.mask_keys:
            m = masks[key]
            if m.shape != (SEGMENTATION_SIZE, SEGMENTATION_SIZE):
                raise InputError(STAGE, "BAD_MASK_SHAPE", f"Mask '{key}' has shape {m.shape}, expected 512x512.")

        bool_masks = {k: np.asarray(masks[k]).astype(bool) for k in self.mask_keys}
        union = np.zeros((SEGMENTATION_SIZE, SEGMENTATION_SIZE), dtype=bool)
        for m in bool_masks.values():
            union |= m
        thoracic_pixels = int(union.sum())
        counts = {k: int(m.sum()) for k, m in bool_masks.items()}

        # ---- Factor 2 (visual) hard stop: independent of metadata ----------
        if thoracic_pixels < self.min_pixels:
            raise ThoraxHardStopError(
                STAGE, "THORACIC_MASS_BELOW_THRESHOLD",
                f"HARD STOP: thoracic pixel mass {thoracic_pixels} < {self.min_pixels} "
                f"(per-structure: {counts}). Input is not a valid chest radiograph.",
                {"thoracic_pixels": thoracic_pixels, "min_pixels": self.min_pixels, "per_structure": counts},
            )

        # ---- Factor 1 (metadata) ------------------------------------------
        normalized = (body_part_examined or "").strip().upper()
        warnings: List[str] = []
        if not normalized:
            if not self.allow_missing_metadata:
                raise ConsensusConflictError(
                    STAGE, "METADATA_BODY_PART_MISSING",
                    "BodyPartExamined (0018,0015) is absent and two-factor consensus is mandatory.",
                )
            metadata_factor, consensus = "MISSING", "VISUAL_ONLY"
            warnings.append("BodyPartExamined absent: routed on visual factor only.")
        elif normalized in self.accepted_body_parts:
            metadata_factor, consensus = "MATCH", "CONFIRMED"
        else:
            raise ConsensusConflictError(
                STAGE, "CONSENSUS_CONFLICT",
                f"DICOM BodyPartExamined='{normalized}' contradicts visual thoracic segmentation "
                f"({thoracic_pixels} px). Possible mislabelled study or wrong-patient image.",
                {"body_part": normalized, "thoracic_pixels": thoracic_pixels},
            )

        return ThoraxRoutingResult(
            passed=True, consensus=consensus, thoracic_pixels=thoracic_pixels,
            left_lung_pixels=counts.get("Left Lung", 0), right_lung_pixels=counts.get("Right Lung", 0),
            heart_pixels=counts.get("Heart", 0), min_pixels=self.min_pixels,
            metadata_body_part=normalized or None, metadata_factor=metadata_factor,
            visual_factor="PASS", warnings=warnings, thoracic_mask=union,
        )
