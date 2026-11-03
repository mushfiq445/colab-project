"""Stage M03 - Abdomen Cross-Organ Veto Router.

LIMITATION NOTE (per Master Prompt Section 2 & 8):
This module is a heuristic, uncalibrated pre-check intended to catch gross
abdominal-only radiographs (excessive sub-diaphragmatic dense mass without
aerated superior lung fields) before they reach the thoracic classifier.
It is NOT clinically calibrated and must not be claimed as a validated classifier.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from src.tier0_errors import OrganRoutingError


@dataclass(frozen=True)
class AbdomenAuditReport:
    subdiaphragmatic_intensity_ratio: float
    max_allowed_ratio: float
    heuristic_uncalibrated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subdiaphragmatic_intensity_ratio": round(self.subdiaphragmatic_intensity_ratio, 4),
            "max_allowed_ratio": self.max_allowed_ratio,
            "heuristic_uncalibrated": self.heuristic_uncalibrated,
        }


class AbdomenRouter:
    """Heuristic cross-organ veto checking for abdominal-dominant attenuation."""

    def __init__(self, *, max_ratio: float = 2.5) -> None:
        self.max_ratio = float(max_ratio)

    def enforce(self, image: np.ndarray) -> AbdomenAuditReport:
        if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
            raise OrganRoutingError(
                "M03", "M03_INVALID_IMAGE", "Abdomen router requires a non-empty 2D numpy array."
            )
        if not np.all(np.isfinite(image)):
            raise OrganRoutingError(
                "M03", "M03_NON_FINITE_IMAGE", "Image contains NaN or infinite values."
            )

        arr = image.astype(np.float32)
        imin, imax = float(np.min(arr)), float(np.max(arr))
        if imax <= imin:
            raise OrganRoutingError(
                "M03", "M03_FLAT_IMAGE", "Image has zero variance (flat pixel array)."
            )

        norm = (arr - imin) / (imax - imin)
        mid = max(1, norm.shape[0] // 2)
        upper_mean = float(np.mean(norm[:mid, :])) + 1e-5
        lower_mean = float(np.mean(norm[mid:, :]))
        ratio = lower_mean / upper_mean

        if ratio > self.max_ratio:
            raise OrganRoutingError(
                "M03",
                "M03_ABDOMEN_DOMINANT_SCAN",
                f"Heuristic sub-diaphragmatic ratio {ratio:.3f} > {self.max_ratio} suggests abdominal scan.",
                {"ratio": ratio, "max_ratio": self.max_ratio},
            )

        return AbdomenAuditReport(
            subdiaphragmatic_intensity_ratio=ratio,
            max_allowed_ratio=self.max_ratio,
            heuristic_uncalibrated=True,
        )
