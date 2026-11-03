"""Stage M01 - DICOM Validator: HIPAA Safe Harbor PHI screening, lossy transfer
syntax rejection, and MONOCHROME1 photometric inversion.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np

from src.tier0_errors import DicomValidationError

# Lossy transfer syntaxes forbidden for primary SaMD diagnostic triage
LOSSY_TRANSFER_SYNTAXES = frozenset({
    "1.2.840.10008.1.2.4.50",  # JPEG Baseline (Process 1)
    "1.2.840.10008.1.2.4.51",  # JPEG Extended (Process 2 & 4)
    "1.2.840.10008.1.2.4.81",  # JPEG-LS Lossy (Near-Lossless)
    "1.2.840.10008.1.2.4.91",  # JPEG 2000 Part 1 (Lossy)
})

# HIPAA Safe Harbor (45 CFR 164.514) tags that must be absent or scrubbed
PHI_TAG_NAMES: Tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "InstitutionName",
    "InstitutionAddress",
    "AccessionNumber",
    "DeviceSerialNumber",
)

SUPPORTED_MODALITIES = frozenset({"CR", "DX", "CT", "MR", "US"})


@dataclass(frozen=True)
class DicomReport:
    file_name: str
    modality: str
    body_part_examined: str
    view_position: str
    rows: int
    columns: int
    bits_stored: int
    photometric_interpretation: str
    inverted_monochrome1: bool
    hipaa_safe_harbor_compliant: bool
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "modality": self.modality,
            "body_part_examined": self.body_part_examined,
            "view_position": self.view_position,
            "rows": self.rows,
            "columns": self.columns,
            "bits_stored": self.bits_stored,
            "photometric_interpretation": self.photometric_interpretation,
            "inverted_monochrome1": self.inverted_monochrome1,
            "hipaa_safe_harbor_compliant": self.hipaa_safe_harbor_compliant,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DicomValidationResult:
    report: DicomReport
    pixel_array: np.ndarray


class DicomValidator:
    """Validates DICOM metadata, rejects PHI and lossy compression, normalises polarity."""

    def __init__(self, *, strict_phi: bool = True, min_bits_stored: int = 8) -> None:
        self.strict_phi = bool(strict_phi)
        self.min_bits_stored = int(min_bits_stored)

    def validate(self, path: Union[str, Path]) -> DicomValidationResult:
        p = Path(path)
        if not p.is_file():
            raise DicomValidationError(
                "M01", "M01_FILE_NOT_FOUND", f"DICOM file not found: {p.name}"
            )
        try:
            import pydicom
        except ImportError as exc:
            raise DicomValidationError(
                "M01", "M01_PYDICOM_UNAVAILABLE", f"pydicom is required for DICOM validation: {exc}"
            ) from exc

        try:
            ds = pydicom.dcmread(str(p), force=False)
        except Exception as exc:
            raise DicomValidationError(
                "M01", "M01_CORRUPT_DICOM", f"Cannot parse DICOM file {p.name}: {exc}"
            ) from exc

        return self.validate_dataset(ds, file_name=p.name)

    def validate_dataset(self, ds: Any, *, file_name: str = "in_memory.dcm") -> DicomValidationResult:
        warnings: List[str] = []

        # 1. Transfer syntax check (reject lossy compression)
        file_meta = getattr(ds, "file_meta", None)
        ts_uid = str(getattr(file_meta, "TransferSyntaxUID", "") or "").strip()
        if ts_uid in LOSSY_TRANSFER_SYNTAXES:
            raise DicomValidationError(
                "M01",
                "M01_LOSSY_TRANSFER_SYNTAX",
                f"Lossy transfer syntax {ts_uid} is prohibited for clinical triage.",
                {"transfer_syntax_uid": ts_uid},
            )
        lossy_flag = str(getattr(ds, "LossyImageCompression", "00") or "00").strip()
        if lossy_flag == "01":
            raise DicomValidationError(
                "M01",
                "M01_LOSSY_COMPRESSION_FLAG",
                "DICOM tag LossyImageCompression=='01' (lossy compressed scan rejected).",
            )

        # 2. Burned-in annotation check
        burned_in = str(getattr(ds, "BurnedInAnnotation", "NO") or "NO").strip().upper()
        if burned_in == "YES":
            raise DicomValidationError(
                "M01",
                "M01_BURNED_IN_ANNOTATION",
                "BurnedInAnnotation=='YES' indicates potential pixel-embedded PHI.",
            )

        # 3. HIPAA Safe Harbor metadata check
        leaked_phi: List[str] = []
        for tag in PHI_TAG_NAMES:
            val = getattr(ds, tag, None)
            if val is not None:
                sval = str(val).strip()
                if sval and sval.upper() not in ("ANONYMIZED", "DEIDENTIFIED", "NONE", "REDACTED"):
                    leaked_phi.append(tag)

        hipaa_ok = len(leaked_phi) == 0
        if not hipaa_ok:
            if self.strict_phi:
                raise DicomValidationError(
                    "M01",
                    "M01_PHI_DETECTED",
                    f"Unscrubbed HIPAA Safe Harbor tags present: {', '.join(leaked_phi)}",
                    {"phi_tags": leaked_phi},
                )
            warnings.append(f"PHI_TAGS_PRESENT:{','.join(leaked_phi)}")

        # 4. Modality & BodyPartExamined checks
        modality = str(getattr(ds, "Modality", "") or "").strip().upper()
        if not modality or modality not in SUPPORTED_MODALITIES:
            raise DicomValidationError(
                "M01",
                "M01_UNSUPPORTED_MODALITY",
                f"Unsupported or missing DICOM Modality {modality!r}.",
                {"modality": modality},
            )

        body_part = str(getattr(ds, "BodyPartExamined", "") or "").strip().upper()
        if not body_part:
            raise DicomValidationError(
                "M01",
                "M01_MISSING_BODY_PART",
                "DICOM tag BodyPartExamined (0018,0015) is missing or empty.",
            )

        view_position = str(getattr(ds, "ViewPosition", "UNKNOWN") or "UNKNOWN").strip().upper()
        bits_stored = int(getattr(ds, "BitsStored", 0) or 0)
        if bits_stored < self.min_bits_stored:
            raise DicomValidationError(
                "M01",
                "M01_INSUFFICIENT_BIT_DEPTH",
                f"BitsStored={bits_stored} is below minimum required {self.min_bits_stored} bits.",
                {"bits_stored": bits_stored},
            )

        # 5. Pixel array extraction & photometric polarity
        try:
            raw_pixels = np.asarray(ds.pixel_array, dtype=np.float32)
        except Exception as exc:
            raise DicomValidationError(
                "M01", "M01_PIXEL_ARRAY_READ_FAILED", f"Failed to decode pixel_array: {exc}"
            ) from exc

        if raw_pixels.ndim != 2 or raw_pixels.size == 0:
            raise DicomValidationError(
                "M01",
                "M01_INVALID_DIMENSIONS",
                f"Expected 2D radiograph pixel array, got shape {raw_pixels.shape}.",
            )
        if not np.all(np.isfinite(raw_pixels)):
            raise DicomValidationError(
                "M01", "M01_NON_FINITE_PIXELS", "Pixel array contains NaN or infinite values."
            )
        if float(np.ptp(raw_pixels)) == 0.0:
            raise DicomValidationError(
                "M01", "M01_BLANK_IMAGE", "Pixel array has zero dynamic range (constant image)."
            )

        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2") or "").strip().upper()
        if photometric not in ("MONOCHROME1", "MONOCHROME2"):
            raise DicomValidationError(
                "M01",
                "M01_UNSUPPORTED_PHOTOMETRIC",
                f"Unsupported PhotometricInterpretation {photometric!r}.",
            )

        inverted = False
        if photometric == "MONOCHROME1":
            raw_pixels = float(np.max(raw_pixels)) - raw_pixels
            inverted = True
            warnings.append("MONOCHROME1_INVERTED_TO_MONOCHROME2")

        rows, cols = int(raw_pixels.shape[0]), int(raw_pixels.shape[1])
        report = DicomReport(
            file_name=file_name,
            modality=modality,
            body_part_examined=body_part,
            view_position=view_position,
            rows=rows,
            columns=cols,
            bits_stored=bits_stored,
            photometric_interpretation=photometric,
            inverted_monochrome1=inverted,
            hipaa_safe_harbor_compliant=hipaa_ok,
            warnings=tuple(warnings),
        )
        return DicomValidationResult(report=report, pixel_array=raw_pixels)
