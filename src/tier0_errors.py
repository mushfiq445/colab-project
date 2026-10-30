"""Shared exception hierarchy for Tier 0 (Pre-Check, Input Validation & Modality Routing).

Every Tier 0 micro-agent fails loudly by raising a subclass of :class:`Tier0Error`.
Each error carries the originating micro-agent (``stage``), a stable machine-readable
``code`` (suitable for the ISO 14971 hazard log / audit trail) and a human-readable
``message`` with the diagnostic detail needed to act on the failure.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class Tier0Error(Exception):
    """Base class for all Tier 0 halting conditions."""

    def __init__(
        self,
        stage: str,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"[{stage}:{code}] {message}")
        self.stage: str = stage
        self.code: str = code
        self.message: str = message
        self.details: Dict[str, Any] = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class InputError(Tier0Error):
    """The input file/array is missing, unreadable or structurally malformed."""


class DicomIntegrityError(Tier0Error):
    """DICOM header or pixel-data integrity failure (M01)."""


class DicomPHIError(Tier0Error):
    """Un-anonymized PHI detected and the PHI policy is 'reject' (M01)."""


class ThoraxHardStopError(Tier0Error):
    """Thoracic pixel mass below the hard-stop threshold (M02)."""


class ConsensusConflictError(Tier0Error):
    """DICOM metadata and visual segmentation disagree (M02)."""


class WrongModalityError(Tier0Error):
    """Input was confirmed to be a non-thoracic (abdominal) image (M03)."""


class BiosignalQualityError(Tier0Error):
    """ECG failed the signal-quality / SNR gate (M04)."""
