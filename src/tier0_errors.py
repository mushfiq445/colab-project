"""Typed Tier 0 pre-check errors with stable hazard codes for ISO 14971 audit logging."""

from __future__ import annotations
from typing import Any, Dict, Optional


class Tier0Error(Exception):
    """Base exception for all Tier 0 pre-check gate failures (always fails closed)."""

    def __init__(
        self,
        stage: str,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.code = str(code)
        self.message = str(message)
        self.details: Dict[str, Any] = dict(details) if details else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class InputError(Tier0Error):
    """Raised when orchestrator input arguments, paths, or devices are invalid."""


class DicomValidationError(Tier0Error):
    """Raised at Stage M01 when DICOM headers, transfer syntax, or PHI rules fail."""


class OrganRoutingError(Tier0Error):
    """Raised at Stage M02 or M03 when anatomical routing or cross-organ gates fail."""


class BiosignalQualityError(Tier0Error):
    """Raised at Stage M04 when 12-lead ECG waveform integrity, lead-off, or SNR fails."""
