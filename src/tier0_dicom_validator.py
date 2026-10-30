"""M01 - DICOM Validator & Integrity Auditor (Tier 0).

Responsibilities
----------------
* Read genuine DICOM Part-10 files with :mod:`pydicom` (no ``force=True`` reads).
* Audit HIPAA Safe Harbor (45 CFR 164.514(b)(2)) de-identification of the direct
  identifier tags PatientName, PatientID, AccessionNumber, InstitutionName and
  ReferringPhysicianName (plus a small set of additional identifier tags).
* Validate pixel bit depth (BitsStored in {12, 14, 16}) and transfer syntax.
* Extract the pixel array, apply the modality rescale, and normalise the photometric
  interpretation so that downstream code always sees MONOCHROME2 semantics
  (higher value == denser / brighter).
* Fail loudly with explicit diagnostics (:mod:`src.tier0_errors`) on any header or
  pixel-integrity problem.

The validator never returns PHI: the result metadata holds only technical fields.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue

from src.tier0_errors import DicomIntegrityError, DicomPHIError, InputError

LOGGER = logging.getLogger(__name__)
STAGE = "M01"

# --------------------------------------------------------------------------- #
# Transfer syntaxes
# --------------------------------------------------------------------------- #
UNCOMPRESSED_TRANSFER_SYNTAXES: Dict[str, str] = {
    "1.2.840.10008.1.2": "Implicit VR Little Endian",
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
    "1.2.840.10008.1.2.1.99": "Deflated Explicit VR Little Endian",
}
LOSSLESS_TRANSFER_SYNTAXES: Dict[str, str] = {
    "1.2.840.10008.1.2.5": "RLE Lossless",
    "1.2.840.10008.1.2.4.57": "JPEG Lossless, Non-Hierarchical",
    "1.2.840.10008.1.2.4.70": "JPEG Lossless, Non-Hierarchical, First-Order Prediction (SV1)",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 Image Compression (Lossless Only)",
}
LOSSY_TRANSFER_SYNTAXES: Dict[str, str] = {
    "1.2.840.10008.1.2.4.50": "JPEG Baseline (Process 1)",
    "1.2.840.10008.1.2.4.51": "JPEG Extended (Process 2 & 4)",
    "1.2.840.10008.1.2.4.91": "JPEG 2000 Image Compression",
}

SUPPORTED_BITS_STORED = (12, 14, 16)
SUPPORTED_MODALITIES = ("CR", "DX")

# --------------------------------------------------------------------------- #
# HIPAA Safe Harbor tag audit
# --------------------------------------------------------------------------- #
REQUIRED_PHI_TAGS = (
    "PatientName",
    "PatientID",
    "AccessionNumber",
    "InstitutionName",
    "ReferringPhysicianName",
)
EXTENDED_PHI_TAGS = (
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientMotherBirthName",
    "InstitutionAddress",
    "PerformingPhysicianName",
    "OperatorsName",
)
# Values that indicate a tag has been scrubbed rather than populated with PHI.
ANONYMIZED_PLACEHOLDERS = frozenset(
    {"", "ANONYMOUS", "ANONYMIZED", "ANONYMISED", "ANON", "REMOVED", "REDACTED",
     "DEIDENTIFIED", "DE-IDENTIFIED", "N/A", "NA", "NONE", "UNKNOWN", "XXX", "^"}
)


@dataclass(frozen=True)
class DicomAuditReport:
    """Serializable (PHI-free) outcome of the M01 audit."""

    file_name: str
    transfer_syntax_uid: str
    transfer_syntax_name: str
    lossy_compressed: bool
    modality: str
    body_part_examined: Optional[str]
    view_position: Optional[str]
    rows: int
    columns: int
    bits_allocated: int
    bits_stored: int
    high_bit: int
    pixel_representation: int
    photometric_interpretation_original: str
    photometric_inverted: bool
    stored_min: float
    stored_max: float
    phi_violations: List[str] = field(default_factory=list)
    hipaa_safe_harbor_compliant: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DicomValidationResult:
    """Pixel data (MONOCHROME2 semantics, float32, shape (H, W)) plus the audit report."""

    pixel_array: np.ndarray
    report: DicomAuditReport


def _tag_text(value: Any) -> str:
    """Flatten any pydicom value (PersonName, MultiValue, str...) to text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, MultiValue)):
        return "\\".join(_tag_text(v) for v in value)
    return str(value).strip()


class DicomValidator:
    """M01 validator.

    Parameters
    ----------
    allow_lossy:
        Accept lossy-compressed transfer syntaxes (flagged with a warning). Default
        ``False``: diagnostic AI must not silently consume lossy pixel data.
    phi_policy:
        ``"flag"`` (default) records PHI findings in the report and logs a warning;
        ``"reject"`` raises :class:`DicomPHIError` on any finding.
    audit_extended_phi:
        Also audit the additional identifier tags in :data:`EXTENDED_PHI_TAGS`.
    """

    def __init__(
        self,
        allow_lossy: bool = False,
        phi_policy: str = "flag",
        audit_extended_phi: bool = True,
    ) -> None:
        if phi_policy not in ("flag", "reject"):
            raise ValueError("phi_policy must be 'flag' or 'reject'")
        self.allow_lossy = allow_lossy
        self.phi_policy = phi_policy
        self.audit_extended_phi = audit_extended_phi

    # ------------------------------------------------------------------ #
    def validate(self, path: Union[str, Path]) -> DicomValidationResult:
        """Run the full M01 audit on ``path``; raises Tier0Error subclasses on failure."""
        file_path = Path(path)
        if not file_path.is_file():
            raise InputError(STAGE, "FILE_NOT_FOUND", f"DICOM file does not exist: {file_path}")

        ds = self._read(file_path)
        warnings: List[str] = []

        ts_uid, ts_name, lossy = self._check_transfer_syntax(ds, warnings)
        modality = self._check_modality(ds)
        self._check_required_tags(ds)
        bits_alloc, bits_stored, high_bit, pixel_rep = self._check_bit_depth(ds)
        photometric = self._check_photometric(ds)
        rows, cols = int(ds.Rows), int(ds.Columns)

        raw = self._extract_pixels(ds, rows, cols)
        self._check_stored_range(raw, bits_stored, pixel_rep)
        stored_min, stored_max = float(raw.min()), float(raw.max())

        image = self._apply_rescale(ds, raw)
        inverted = photometric == "MONOCHROME1"
        if inverted:
            image = (image.max() + image.min()) - image
            LOGGER.info("M01: MONOCHROME1 input inverted to MONOCHROME2 semantics")
        image = np.ascontiguousarray(image, dtype=np.float32)

        if not np.isfinite(image).all():
            raise DicomIntegrityError(STAGE, "NON_FINITE_PIXELS", "Pixel array contains NaN/Inf after rescale.")
        if float(np.ptp(image)) <= 0.0:
            raise DicomIntegrityError(
                STAGE, "CONSTANT_IMAGE",
                "Pixel array is constant (zero dynamic range); image carries no diagnostic information.",
            )

        phi_violations = self._audit_phi(ds)
        if phi_violations:
            LOGGER.warning("M01: un-anonymized PHI tags present: %s", ", ".join(phi_violations))
            if self.phi_policy == "reject":
                raise DicomPHIError(
                    STAGE, "PHI_DETECTED",
                    f"HIPAA Safe Harbor violation, un-anonymized tags: {', '.join(phi_violations)}",
                    {"tags": phi_violations},
                )
            warnings.append("PHI_PRESENT: " + ", ".join(phi_violations))

        report = DicomAuditReport(
            file_name=file_path.name,
            transfer_syntax_uid=ts_uid,
            transfer_syntax_name=ts_name,
            lossy_compressed=lossy,
            modality=modality,
            body_part_examined=(_tag_text(ds.get("BodyPartExamined")) or None),
            view_position=(_tag_text(ds.get("ViewPosition")) or None),
            rows=rows,
            columns=cols,
            bits_allocated=bits_alloc,
            bits_stored=bits_stored,
            high_bit=high_bit,
            pixel_representation=pixel_rep,
            photometric_interpretation_original=photometric,
            photometric_inverted=inverted,
            stored_min=stored_min,
            stored_max=stored_max,
            phi_violations=phi_violations,
            hipaa_safe_harbor_compliant=not phi_violations,
            warnings=warnings,
        )
        return DicomValidationResult(pixel_array=image, report=report)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read(path: Path) -> pydicom.dataset.FileDataset:
        try:
            return pydicom.dcmread(str(path), force=False)
        except InvalidDicomError as exc:
            raise DicomIntegrityError(
                STAGE, "INVALID_DICOM", f"Not a valid DICOM Part-10 file (missing preamble/'DICM' prefix): {exc}"
            ) from exc
        except Exception as exc:  # truncated/corrupt header
            raise DicomIntegrityError(STAGE, "HEADER_READ_FAILED", f"Failed to parse DICOM header: {exc!r}") from exc

    def _check_transfer_syntax(self, ds: Any, warnings: List[str]) -> "tuple[str, str, bool]":
        meta = getattr(ds, "file_meta", None)
        uid = str(getattr(meta, "TransferSyntaxUID", "") or "")
        if not uid:
            raise DicomIntegrityError(STAGE, "MISSING_TRANSFER_SYNTAX", "file_meta.TransferSyntaxUID (0002,0010) is absent.")
        if uid in UNCOMPRESSED_TRANSFER_SYNTAXES:
            return uid, UNCOMPRESSED_TRANSFER_SYNTAXES[uid], False
        if uid in LOSSLESS_TRANSFER_SYNTAXES:
            return uid, LOSSLESS_TRANSFER_SYNTAXES[uid], False
        if uid in LOSSY_TRANSFER_SYNTAXES:
            name = LOSSY_TRANSFER_SYNTAXES[uid]
            if not self.allow_lossy:
                raise DicomIntegrityError(
                    STAGE, "LOSSY_TRANSFER_SYNTAX",
                    f"Lossy transfer syntax rejected: {name} ({uid}). Supply an uncompressed or lossless study.",
                    {"transfer_syntax_uid": uid},
                )
            warnings.append(f"LOSSY_COMPRESSION: {name}")
            return uid, name, True
        raise DicomIntegrityError(
            STAGE, "UNSUPPORTED_TRANSFER_SYNTAX", f"Unsupported transfer syntax UID: {uid}", {"transfer_syntax_uid": uid}
        )

    @staticmethod
    def _check_modality(ds: Any) -> str:
        modality = _tag_text(ds.get("Modality")).upper()
        if modality not in SUPPORTED_MODALITIES:
            raise DicomIntegrityError(
                STAGE, "UNSUPPORTED_DICOM_MODALITY",
                f"Modality '{modality or 'MISSING'}' is not a projection radiograph (expected one of {SUPPORTED_MODALITIES}).",
            )
        return modality

    @staticmethod
    def _check_required_tags(ds: Any) -> None:
        required = (
            "Rows", "Columns", "BitsAllocated", "BitsStored", "HighBit",
            "PixelRepresentation", "SamplesPerPixel", "PhotometricInterpretation",
        )
        missing = [t for t in required if t not in ds]
        if missing:
            raise DicomIntegrityError(STAGE, "MISSING_REQUIRED_TAGS", f"Required image-pixel tags missing: {missing}", {"missing": missing})
        if "PixelData" not in ds:
            raise DicomIntegrityError(STAGE, "MISSING_PIXEL_DATA", "PixelData (7FE0,0010) is absent.")
        if int(ds.SamplesPerPixel) != 1:
            raise DicomIntegrityError(STAGE, "NOT_SINGLE_CHANNEL", f"SamplesPerPixel={int(ds.SamplesPerPixel)}; expected 1 (grayscale).")
        frames = int(ds.get("NumberOfFrames", 1) or 1)
        if frames != 1:
            raise DicomIntegrityError(STAGE, "MULTI_FRAME", f"NumberOfFrames={frames}; only single-frame radiographs are supported.")
        if int(ds.Rows) < 1 or int(ds.Columns) < 1:
            raise DicomIntegrityError(STAGE, "INVALID_DIMENSIONS", f"Invalid Rows/Columns: {ds.Rows}x{ds.Columns}.")

    @staticmethod
    def _check_bit_depth(ds: Any) -> "tuple[int, int, int, int]":
        bits_alloc, bits_stored = int(ds.BitsAllocated), int(ds.BitsStored)
        high_bit, pixel_rep = int(ds.HighBit), int(ds.PixelRepresentation)
        if bits_alloc != 16:
            raise DicomIntegrityError(STAGE, "UNSUPPORTED_BITS_ALLOCATED", f"BitsAllocated={bits_alloc}; expected 16 for 12/14/16-bit data.")
        if bits_stored not in SUPPORTED_BITS_STORED:
            raise DicomIntegrityError(
                STAGE, "UNSUPPORTED_BIT_DEPTH", f"BitsStored={bits_stored}; supported depths are {SUPPORTED_BITS_STORED}."
            )
        if high_bit != bits_stored - 1:
            raise DicomIntegrityError(
                STAGE, "INCONSISTENT_HIGH_BIT", f"HighBit={high_bit} inconsistent with BitsStored={bits_stored} (expected {bits_stored - 1})."
            )
        if pixel_rep not in (0, 1):
            raise DicomIntegrityError(STAGE, "INVALID_PIXEL_REPRESENTATION", f"PixelRepresentation={pixel_rep}; expected 0 or 1.")
        return bits_alloc, bits_stored, high_bit, pixel_rep

    @staticmethod
    def _check_photometric(ds: Any) -> str:
        photometric = _tag_text(ds.PhotometricInterpretation).upper()
        if photometric not in ("MONOCHROME1", "MONOCHROME2"):
            raise DicomIntegrityError(
                STAGE, "UNSUPPORTED_PHOTOMETRIC_INTERPRETATION",
                f"PhotometricInterpretation='{photometric}'; only MONOCHROME1/MONOCHROME2 are supported.",
            )
        return photometric

    @staticmethod
    def _extract_pixels(ds: Any, rows: int, cols: int) -> np.ndarray:
        try:
            arr = np.asarray(ds.pixel_array)
        except Exception as exc:
            raise DicomIntegrityError(
                STAGE, "PIXEL_DECODE_FAILED",
                f"Pixel data could not be decoded (truncated buffer, corrupt stream or missing codec plugin): {exc!r}",
            ) from exc
        if arr.ndim != 2 or arr.shape != (rows, cols):
            raise DicomIntegrityError(
                STAGE, "PIXEL_SHAPE_MISMATCH", f"Decoded pixel array shape {arr.shape} != header (Rows, Columns)=({rows}, {cols})."
            )
        if not np.issubdtype(arr.dtype, np.integer):
            raise DicomIntegrityError(STAGE, "NON_INTEGER_PIXELS", f"Decoded pixel dtype {arr.dtype} is not integer.")
        return arr

    @staticmethod
    def _check_stored_range(arr: np.ndarray, bits_stored: int, pixel_rep: int) -> None:
        lo, hi = int(arr.min()), int(arr.max())
        if pixel_rep == 0:
            lim_lo, lim_hi = 0, (1 << bits_stored) - 1
        else:
            lim_lo, lim_hi = -(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1
        if lo < lim_lo or hi > lim_hi:
            raise DicomIntegrityError(
                STAGE, "PIXEL_RANGE_VIOLATION",
                f"Pixel values [{lo}, {hi}] exceed the {bits_stored}-bit "
                f"{'signed' if pixel_rep else 'unsigned'} range [{lim_lo}, {lim_hi}] (corrupt data or wrong BitsStored).",
            )

    @staticmethod
    def _apply_rescale(ds: Any, arr: np.ndarray) -> np.ndarray:
        slope = float(ds.get("RescaleSlope", 1.0) or 1.0)
        intercept = float(ds.get("RescaleIntercept", 0.0) or 0.0)
        if not (np.isfinite(slope) and np.isfinite(intercept)) or slope == 0.0:
            raise DicomIntegrityError(STAGE, "INVALID_RESCALE", f"Invalid RescaleSlope/Intercept: {slope}/{intercept}.")
        return arr.astype(np.float64) * slope + intercept

    def _audit_phi(self, ds: Any) -> List[str]:
        tags = list(REQUIRED_PHI_TAGS) + (list(EXTENDED_PHI_TAGS) if self.audit_extended_phi else [])
        violations: List[str] = []
        for name in tags:
            if name not in ds:
                continue
            text = _tag_text(ds.get(name)).upper()
            if text in ANONYMIZED_PLACEHOLDERS:
                continue
            violations.append(name)
        return violations


def validate_dicom(path: Union[str, Path], **kwargs: Any) -> DicomValidationResult:
    """Functional convenience wrapper around :class:`DicomValidator`."""
    return DicomValidator(**kwargs).validate(path)
