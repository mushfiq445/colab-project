"""M03 - Abdomen Visual Router (Tier 0).

Lightweight, deterministic spatial radiolucency analyzer that catches wrong-modality
(abdominal) radiographs *before* thoracic inference. No learned weights are required.

Input convention: 2-D image with MONOCHROME2 semantics (higher == denser), i.e. the
output of M01. Features are computed on a 256 x 256 central region of interest:

* ``upper_aeration`` / ``lower_aeration`` - fraction of Otsu-defined radiolucent
  (air-density) pixels in the upper / lower half (mean of left & right quadrants).
* ``density_ratio`` - mean upper-half density divided by mean lower-half density.
* ``aeration_gradient`` - lower minus upper aeration.

A chest radiograph shows large bilateral lucent lung fields in the *upper* half
(high upper aeration, upper denser-ratio < 1). An abdominal radiograph is dense in the
upper half (liver/diaphragm) with lucent bowel gas distributed *below*. Three sub-scores
in [0, 1] are combined into a routing confidence. The weights and breakpoints in
:class:`AbdomenRouterConfig` are heuristic defaults and MUST be calibrated against a
labelled validation set (ROC / operating-point analysis) before clinical release.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from scipy import ndimage

from src.tier0_errors import InputError, WrongModalityError

LOGGER = logging.getLogger(__name__)
STAGE = "M03"
ANALYSIS_SIZE = 256


@dataclass(frozen=True)
class AbdomenRouterConfig:
    confidence_threshold: float = 0.60  # >= -> abdomen confirmed
    upper_aeration_floor: float = 0.25  # thorax expects >= this in upper half
    density_ratio_lo: float = 0.85
    density_ratio_span: float = 0.30
    gradient_span: float = 0.15
    w_upper_aeration: float = 0.40
    w_density: float = 0.30
    w_gradient: float = 0.30
    roi_rows: tuple = (0.05, 0.95)
    roi_cols: tuple = (0.15, 0.85)


@dataclass(frozen=True)
class AbdomenRoutingResult:
    abdomen_confidence: float
    is_abdomen: bool
    threshold: float
    features: Dict[str, float] = field(default_factory=dict)
    sub_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abdomen_confidence": self.abdomen_confidence, "is_abdomen": self.is_abdomen,
            "threshold": self.threshold, "features": dict(self.features),
            "sub_scores": dict(self.sub_scores),
        }


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centers)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (w0 * w1) * ((m0 / w0) - ((mt - m0) / w1)) ** 2
    between = np.nan_to_num(between, nan=0.0, posinf=0.0, neginf=0.0)
    return float(centers[int(np.argmax(between))])


class AbdomenRouter:
    """M03 router."""

    def __init__(self, config: AbdomenRouterConfig = AbdomenRouterConfig()) -> None:
        self.config = config

    def assess(self, image: np.ndarray) -> AbdomenRoutingResult:
        cfg = self.config
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 2 or min(img.shape) < 32:
            raise InputError(STAGE, "BAD_IMAGE", f"Expected 2-D image >= 32x32, got shape {img.shape}.")
        if not np.isfinite(img).all():
            raise InputError(STAGE, "NON_FINITE_PIXELS", "Image contains NaN/Inf.")

        small = ndimage.zoom(img, (ANALYSIS_SIZE / img.shape[0], ANALYSIS_SIZE / img.shape[1]), order=1, mode="nearest")
        lo, hi = np.percentile(small, [1.0, 99.0])
        if hi - lo <= 1e-9:
            raise InputError(STAGE, "NO_DYNAMIC_RANGE", "Image has no usable dynamic range for radiolucency analysis.")
        unit = np.clip((small - lo) / (hi - lo), 0.0, 1.0)

        r0, r1 = (int(f * unit.shape[0]) for f in cfg.roi_rows)
        c0, c1 = (int(f * unit.shape[1]) for f in cfg.roi_cols)
        roi = unit[r0:r1, c0:c1]
        t = _otsu_threshold(roi.ravel())
        lucent = roi < t

        mid_r, mid_c = roi.shape[0] // 2, roi.shape[1] // 2
        quad = {
            "UL": lucent[:mid_r, :mid_c].mean(), "UR": lucent[:mid_r, mid_c:].mean(),
            "LL": lucent[mid_r:, :mid_c].mean(), "LR": lucent[mid_r:, mid_c:].mean(),
        }
        upper_aer = 0.5 * (quad["UL"] + quad["UR"])
        lower_aer = 0.5 * (quad["LL"] + quad["LR"])
        upper_density = float(roi[:mid_r].mean())
        lower_density = float(roi[mid_r:].mean())
        density_ratio = upper_density / max(lower_density, 1e-6)
        gradient = lower_aer - upper_aer

        s_upper = float(np.clip((cfg.upper_aeration_floor - upper_aer) / cfg.upper_aeration_floor, 0.0, 1.0))
        s_density = float(np.clip((density_ratio - cfg.density_ratio_lo) / cfg.density_ratio_span, 0.0, 1.0))
        s_grad = float(np.clip(gradient / cfg.gradient_span, 0.0, 1.0))
        wsum = cfg.w_upper_aeration + cfg.w_density + cfg.w_gradient
        confidence = float(
            (cfg.w_upper_aeration * s_upper + cfg.w_density * s_density + cfg.w_gradient * s_grad) / wsum
        )

        return AbdomenRoutingResult(
            abdomen_confidence=confidence,
            is_abdomen=confidence >= cfg.confidence_threshold,
            threshold=cfg.confidence_threshold,
            features={
                "otsu_threshold": float(t),
                **{f"aeration_{k}": float(v) for k, v in quad.items()},
                "upper_aeration": float(upper_aer), "lower_aeration": float(lower_aer),
                "upper_density": upper_density, "lower_density": lower_density,
                "density_ratio": float(density_ratio), "aeration_gradient": float(gradient),
            },
            sub_scores={"upper_aeration_deficit": s_upper, "density_inversion": s_density, "lower_gas_dominance": s_grad},
        )

    def enforce(self, image: np.ndarray) -> AbdomenRoutingResult:
        """Assess and lock out downstream thoracic inference if abdomen is confirmed."""
        result = self.assess(image)
        if result.is_abdomen:
            raise WrongModalityError(
                STAGE, "ABDOMEN_CONFIRMED",
                f"Abdominal features confirmed (routing confidence {result.abdomen_confidence:.3f} >= "
                f"{result.threshold:.2f}); thoracic inference is locked out.",
                result.to_dict(),
            )
        return result
