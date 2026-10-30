"""Unit & integration tests for Tier 0 (M01-M04 + orchestrator).

Run from the repository root:  python -m unittest tests.test_tier0 -v

Fixtures are generated on the fly: DICOM files are written with pydicom into a temporary
directory and ECG waveforms are synthesised at 500 Hz. The segmentation backend is replaced
by a deterministic geometric engine (``GeometricSegmentationEngine``) so the M02 pixel-mass
logic is verified without downloading model weights; the production ``XRVPSPNetEngine``
is not exercised by unit tests. DICOM/orchestrator tests are skipped (not failed) when
pydicom / torch are not installed.
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tier0_abdomen_router import AbdomenRouter
from src.tier0_biosignal_gate import BiosignalGate, load_ecg
from src.tier0_errors import (
    BiosignalQualityError, ConsensusConflictError, DicomIntegrityError, DicomPHIError,
    InputError, ThoraxHardStopError, WrongModalityError,
)
from src.tier0_thorax_router import SEGMENTATION_SIZE, ThoraxRouter

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from src.tier0_dicom_validator import DicomValidator
    HAVE_PYDICOM = True
except ImportError:  # pragma: no cover
    HAVE_PYDICOM = False

try:
    import torch
    from src.router import execute_tier0_precheck
    HAVE_ROUTER = HAVE_PYDICOM
except ImportError:  # pragma: no cover
    HAVE_ROUTER = False

FS = 500


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def chest_like(n: int = 512, seed: int = 0) -> np.ndarray:
    """Radiograph-like image (0..1, higher == denser): bilateral lucent lungs in the upper half."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.full((n, n), 0.62)
    for cx in (0.34, 0.66):
        img[((xx - cx) / 0.13) ** 2 + ((yy - 0.38) / 0.25) ** 2 < 1.0] = 0.15
    img[yy > 0.75] = 0.72
    return np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)


def abdomen_like(n: int = 512, seed: int = 0) -> np.ndarray:
    """Dense upper half (liver/diaphragm) with lucent bowel-gas pockets in the lower half."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.full((n, n), 0.60)
    img[yy < 0.5] = 0.72
    for cx, cy in ((0.32, 0.62), (0.55, 0.70), (0.45, 0.82), (0.68, 0.60)):
        img[((xx - cx) / 0.09) ** 2 + ((yy - cy) / 0.07) ** 2 < 1.0] = 0.12
    return np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)


def synthetic_ecg(duration_s: float = 10.0, noise_sd: float = 0.01, hr: float = 72.0, seed: int = 0) -> np.ndarray:
    """12-lead PQRST waveform, shape (12, n), mV, 500 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * FS)) / FS
    sig = np.zeros_like(t)
    for bt in np.arange(0.4, duration_s, 60.0 / hr):
        d = t - bt
        sig += 0.12 * np.exp(-((d + 0.20) ** 2) / (2 * 0.025 ** 2))
        sig += -0.15 * np.exp(-((d + 0.04) ** 2) / (2 * 0.010 ** 2))
        sig += 1.00 * np.exp(-(d ** 2) / (2 * 0.012 ** 2))
        sig += -0.25 * np.exp(-((d - 0.04) ** 2) / (2 * 0.012 ** 2))
        sig += 0.30 * np.exp(-((d - 0.25) ** 2) / (2 * 0.050 ** 2))
    gains = np.array([1.0, 1.2, 0.6, -0.8, 0.5, 0.9, 0.4, 0.8, 1.1, 1.3, 1.0, 0.8])
    return gains[:, None] * sig[None, :] + rng.normal(0, noise_sd, (12, t.size))


class GeometricSegmentationEngine:
    """Deterministic test backend: emits masks with exact, caller-chosen pixel counts."""

    def __init__(self, left: int, right: int, heart: int) -> None:
        self.counts = {"Left Lung": left, "Right Lung": right, "Heart": heart}
        self.offsets = {"Left Lung": 0, "Right Lung": 100_000, "Heart": 200_000}

    def predict_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        assert image.shape == (SEGMENTATION_SIZE, SEGMENTATION_SIZE) and image.dtype == np.float32
        assert image.min() >= -1024.0 - 1e-3 and image.max() <= 1024.0 + 1e-3
        out = {}
        for key, n in self.counts.items():
            flat = np.zeros(SEGMENTATION_SIZE * SEGMENTATION_SIZE, dtype=bool)
            flat[self.offsets[key]: self.offsets[key] + n] = True
            out[key] = flat.reshape(SEGMENTATION_SIZE, SEGMENTATION_SIZE)
        out["Spine"] = np.zeros((SEGMENTATION_SIZE, SEGMENTATION_SIZE), dtype=bool)
        return out


def write_dicom(
    path: Path, pixels: np.ndarray, *, bits_stored: int = 12, photometric: str = "MONOCHROME2",
    body_part: Optional[str] = "CHEST", phi: bool = False, modality: str = "DX", pixel_rep: int = 0,
) -> Path:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    if phi:
        ds.PatientName, ds.PatientID = "Doe^Jane", "MRN123456"
        ds.AccessionNumber, ds.InstitutionName = "ACC987654", "General Hospital"
        ds.ReferringPhysicianName = "Smith^John"
    else:
        ds.PatientName, ds.PatientID = "ANONYMOUS", "ANONYMOUS"
        ds.AccessionNumber, ds.InstitutionName, ds.ReferringPhysicianName = "", "REMOVED", ""
    if body_part is not None:
        ds.BodyPartExamined = body_part
    ds.ViewPosition = "PA"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, bits_stored, bits_stored - 1
    ds.PixelRepresentation = pixel_rep
    dtype = np.int16 if pixel_rep else np.uint16
    ds.PixelData = pixels.astype(dtype).tobytes()
    ds["PixelData"].VR = "OW"
    if int(pydicom.__version__.split(".")[0]) < 3:
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(str(path), write_like_original=False)
    else:
        ds.save_as(str(path), enforce_file_format=True)
    return path


def to_counts(img01: np.ndarray, bits: int) -> np.ndarray:
    return np.round(img01 * ((1 << bits) - 1)).astype(np.uint16)


# --------------------------------------------------------------------------- #
# M01
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_PYDICOM, "pydicom not installed")
class TestM01DicomValidator(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.validator = DicomValidator()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_supported_bit_depths(self) -> None:
        for bits in (12, 14, 16):
            with self.subTest(bits=bits):
                path = write_dicom(self.dir / f"b{bits}.dcm", to_counts(chest_like(128), bits), bits_stored=bits)
                res = self.validator.validate(path)
                self.assertEqual(res.report.bits_stored, bits)
                self.assertEqual(res.pixel_array.shape, (128, 128))
                self.assertEqual(res.pixel_array.dtype, np.float32)
                self.assertEqual(res.report.transfer_syntax_uid, "1.2.840.10008.1.2.1")

    def test_unsupported_bit_depth_rejected(self) -> None:
        path = write_dicom(self.dir / "b10.dcm", (chest_like(64) * 1000).astype(np.uint16), bits_stored=10)
        with self.assertRaises(DicomIntegrityError) as ctx:
            self.validator.validate(path)
        self.assertEqual(ctx.exception.code, "UNSUPPORTED_BIT_DEPTH")

    def test_pixel_range_violation(self) -> None:
        px = to_counts(chest_like(64), 12)
        px[0, 0] = 5000  # > 4095
        path = write_dicom(self.dir / "range.dcm", px, bits_stored=12)
        with self.assertRaises(DicomIntegrityError) as ctx:
            self.validator.validate(path)
        self.assertEqual(ctx.exception.code, "PIXEL_RANGE_VIOLATION")

    def test_phi_tags_flagged(self) -> None:
        path = write_dicom(self.dir / "phi.dcm", to_counts(chest_like(64), 12), phi=True)
        res = self.validator.validate(path)
        self.assertFalse(res.report.hipaa_safe_harbor_compliant)
        for tag in ("PatientName", "PatientID", "AccessionNumber", "InstitutionName", "ReferringPhysicianName"):
            self.assertIn(tag, res.report.phi_violations)

    def test_phi_reject_policy(self) -> None:
        path = write_dicom(self.dir / "phi2.dcm", to_counts(chest_like(64), 12), phi=True)
        with self.assertRaises(DicomPHIError):
            DicomValidator(phi_policy="reject").validate(path)

    def test_anonymized_file_clean(self) -> None:
        path = write_dicom(self.dir / "anon.dcm", to_counts(chest_like(64), 12), phi=False)
        res = self.validator.validate(path)
        self.assertTrue(res.report.hipaa_safe_harbor_compliant)
        self.assertEqual(res.report.phi_violations, [])

    def test_monochrome1_is_inverted(self) -> None:
        grad = np.tile(np.linspace(0, 4095, 64).astype(np.uint16), (64, 1))
        r2 = self.validator.validate(write_dicom(self.dir / "m2.dcm", grad, photometric="MONOCHROME2"))
        r1 = self.validator.validate(write_dicom(self.dir / "m1.dcm", grad, photometric="MONOCHROME1"))
        self.assertFalse(r2.report.photometric_inverted)
        self.assertTrue(r1.report.photometric_inverted)
        np.testing.assert_allclose(r1.pixel_array, (grad.max() + grad.min()) - r2.pixel_array, atol=1e-3)

    def test_unsupported_photometric(self) -> None:
        path = write_dicom(self.dir / "pal.dcm", to_counts(chest_like(64), 12), photometric="PALETTE COLOR")
        with self.assertRaises(DicomIntegrityError) as ctx:
            self.validator.validate(path)
        self.assertEqual(ctx.exception.code, "UNSUPPORTED_PHOTOMETRIC_INTERPRETATION")

    def test_wrong_dicom_modality(self) -> None:
        path = write_dicom(self.dir / "ct.dcm", to_counts(chest_like(64), 12), modality="CT")
        with self.assertRaises(DicomIntegrityError) as ctx:
            self.validator.validate(path)
        self.assertEqual(ctx.exception.code, "UNSUPPORTED_DICOM_MODALITY")

    def test_constant_image_rejected(self) -> None:
        path = write_dicom(self.dir / "const.dcm", np.full((64, 64), 1000, dtype=np.uint16))
        with self.assertRaises(DicomIntegrityError) as ctx:
            self.validator.validate(path)
        self.assertEqual(ctx.exception.code, "CONSTANT_IMAGE")

    def test_transfer_syntax_policy(self) -> None:
        def ds_with(uid: str) -> types.SimpleNamespace:
            return types.SimpleNamespace(file_meta=types.SimpleNamespace(TransferSyntaxUID=uid))

        with self.assertRaises(DicomIntegrityError) as ctx:
            self.validator._check_transfer_syntax(ds_with("1.2.840.10008.1.2.4.50"), [])
        self.assertEqual(ctx.exception.code, "LOSSY_TRANSFER_SYNTAX")
        with self.assertRaises(DicomIntegrityError) as ctx:
            self.validator._check_transfer_syntax(ds_with("1.2.840.10008.1.2.2"), [])
        self.assertEqual(ctx.exception.code, "UNSUPPORTED_TRANSFER_SYNTAX")
        warnings: list = []
        uid, _, lossy = DicomValidator(allow_lossy=True)._check_transfer_syntax(ds_with("1.2.840.10008.1.2.4.91"), warnings)
        self.assertTrue(lossy)
        self.assertTrue(warnings)

    def test_truncated_pixel_data_fails_loudly(self) -> None:
        path = write_dicom(self.dir / "full.dcm", to_counts(chest_like(128), 12))
        data = path.read_bytes()
        bad = self.dir / "trunc.dcm"
        bad.write_bytes(data[: len(data) - 4000])
        with self.assertRaises(DicomIntegrityError):
            self.validator.validate(bad)

    def test_not_a_dicom_file(self) -> None:
        bad = self.dir / "junk.dcm"
        bad.write_bytes(b"this is definitely not DICOM" * 50)
        with self.assertRaises(DicomIntegrityError) as ctx:
            self.validator.validate(bad)
        self.assertEqual(ctx.exception.code, "INVALID_DICOM")

    def test_missing_file(self) -> None:
        with self.assertRaises(InputError):
            self.validator.validate(self.dir / "nope.dcm")


# --------------------------------------------------------------------------- #
# M02
# --------------------------------------------------------------------------- #
class TestM02ThoraxRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.image = chest_like(512).astype(np.float32)

    def test_pass_at_or_above_threshold(self) -> None:
        for left, right, heart in ((1200, 1200, 0), (500, 500, 0), (1000, 0, 0)):
            with self.subTest(total=left + right + heart):
                res = ThoraxRouter(GeometricSegmentationEngine(left, right, heart)).audit(self.image, "CHEST")
                self.assertTrue(res.passed)
                self.assertEqual(res.thoracic_pixels, left + right + heart)
                self.assertEqual(res.consensus, "CONFIRMED")

    def test_hard_stop_below_threshold(self) -> None:
        for counts in ((999, 0, 0), (300, 300, 300), (0, 0, 0)):
            with self.subTest(counts=counts):
                with self.assertRaises(ThoraxHardStopError) as ctx:
                    ThoraxRouter(GeometricSegmentationEngine(*counts)).audit(self.image, "CHEST")
                self.assertEqual(ctx.exception.code, "THORACIC_MASS_BELOW_THRESHOLD")

    def test_hard_stop_overrides_matching_metadata(self) -> None:
        with self.assertRaises(ThoraxHardStopError):
            ThoraxRouter(GeometricSegmentationEngine(10, 10, 10)).audit(self.image, "THORAX")

    def test_metadata_conflict_halts(self) -> None:
        with self.assertRaises(ConsensusConflictError) as ctx:
            ThoraxRouter(GeometricSegmentationEngine(5000, 5000, 0)).audit(self.image, "ABDOMEN")
        self.assertEqual(ctx.exception.code, "CONSENSUS_CONFLICT")

    def test_missing_metadata_policy(self) -> None:
        engine = GeometricSegmentationEngine(5000, 5000, 0)
        res = ThoraxRouter(engine).audit(self.image, None)
        self.assertEqual(res.consensus, "VISUAL_ONLY")
        self.assertTrue(res.warnings)
        with self.assertRaises(ConsensusConflictError):
            ThoraxRouter(engine, allow_missing_metadata=False).audit(self.image, "")

    def test_body_part_normalisation(self) -> None:
        res = ThoraxRouter(GeometricSegmentationEngine(5000, 5000, 0)).audit(self.image, " thorax ")
        self.assertEqual(res.metadata_body_part, "THORAX")


# --------------------------------------------------------------------------- #
# M03
# --------------------------------------------------------------------------- #
class TestM03AbdomenRouter(unittest.TestCase):
    def test_chest_passes(self) -> None:
        res = AbdomenRouter().enforce(chest_like(512))
        self.assertFalse(res.is_abdomen)
        self.assertLess(res.abdomen_confidence, 0.30)
        self.assertGreater(res.features["upper_aeration"], res.features["lower_aeration"])

    def test_abdomen_lockout(self) -> None:
        router = AbdomenRouter()
        res = router.assess(abdomen_like(512))
        self.assertTrue(res.is_abdomen)
        self.assertGreaterEqual(res.abdomen_confidence, 0.90)
        with self.assertRaises(WrongModalityError) as ctx:
            router.enforce(abdomen_like(512))
        self.assertEqual(ctx.exception.code, "ABDOMEN_CONFIRMED")
        self.assertIn("abdomen_confidence", ctx.exception.details)

    def test_confidence_bounded_and_resolution_invariant(self) -> None:
        a = AbdomenRouter().assess(abdomen_like(512)).abdomen_confidence
        b = AbdomenRouter().assess(abdomen_like(1024)).abdomen_confidence
        self.assertTrue(0.0 <= a <= 1.0 and 0.0 <= b <= 1.0)
        self.assertAlmostEqual(a, b, delta=0.1)

    def test_degenerate_inputs(self) -> None:
        with self.assertRaises(InputError):
            AbdomenRouter().assess(np.full((256, 256), 0.5))
        with self.assertRaises(InputError):
            AbdomenRouter().assess(np.zeros((8, 8)))
        bad = chest_like(256)
        bad[0, 0] = np.nan
        with self.assertRaises(InputError):
            AbdomenRouter().assess(bad)


# --------------------------------------------------------------------------- #
# M04
# --------------------------------------------------------------------------- #
class TestM04BiosignalGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = BiosignalGate()

    def test_clean_ecg_passes(self) -> None:
        res = self.gate.enforce(synthetic_ecg(noise_sd=0.01))
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.snr_db, 6.0)
        self.assertGreaterEqual(res.min_lead_snr_db, 6.0)
        self.assertEqual(res.filtered_mv.shape, (12, 10 * FS))

    def test_samples_by_leads_orientation_accepted(self) -> None:
        self.assertTrue(self.gate.evaluate(synthetic_ecg().T).passed)

    def test_flatline_rejected(self) -> None:
        res = self.gate.evaluate(np.zeros((12, 10 * FS)))
        self.assertFalse(res.passed)
        self.assertTrue(all("FLATLINE" in l.reasons for l in res.leads))
        with self.assertRaises(BiosignalQualityError):
            self.gate.enforce(np.zeros((12, 10 * FS)))

    def test_near_flatline_below_ptp_threshold(self) -> None:
        rng = np.random.default_rng(1)
        ecg = rng.normal(0, 0.003, (12, 10 * FS))  # ptp ~ 0.02 mV < 0.05 mV
        res = self.gate.evaluate(ecg)
        self.assertFalse(res.passed)
        self.assertTrue(all("FLATLINE" in l.reasons and "DISCONNECTED" not in l.reasons for l in res.leads))

    def test_ptp_just_above_threshold_not_flatline(self) -> None:
        ecg = synthetic_ecg(noise_sd=0.0)
        ecg[6] = 0.06 * np.sin(2 * np.pi * 1.2 * np.arange(ecg.shape[1]) / FS)  # ptp = 0.12 mV
        res = self.gate.evaluate(ecg)
        self.assertNotIn("FLATLINE", res.leads[6].reasons)

    def test_high_frequency_noise_rejected(self) -> None:
        res = self.gate.evaluate(synthetic_ecg(noise_sd=1.0, seed=3))
        self.assertFalse(res.passed)
        self.assertLess(res.snr_db, 6.0)
        self.assertTrue(any(r.endswith("LOW_SNR") for r in res.failure_reasons))
        with self.assertRaises(BiosignalQualityError) as ctx:
            self.gate.enforce(synthetic_ecg(noise_sd=1.0, seed=3))
        self.assertEqual(ctx.exception.code, "SIGNAL_QUALITY_REJECTED")

    def test_powerline_interference_rejected(self) -> None:
        ecg = synthetic_ecg(noise_sd=0.01)
        t = np.arange(ecg.shape[1]) / FS
        ecg += 1.5 * np.sin(2 * np.pi * 50.0 * t)[None, :]
        res = self.gate.evaluate(ecg)
        self.assertFalse(res.passed)
        self.assertTrue(any(r.endswith("LOW_SNR") for r in res.failure_reasons))

    def test_disconnected_lead(self) -> None:
        ecg = synthetic_ecg()
        ecg[4] = 0.5  # lead-off: constant value
        res = self.gate.evaluate(ecg)
        self.assertFalse(res.passed)
        self.assertIn("DISCONNECTED", res.leads[4].reasons)
        ecg2 = synthetic_ecg()
        ecg2[2, 100:200] = np.nan
        self.assertIn("DISCONNECTED", self.gate.evaluate(ecg2).leads[2].reasons)

    def test_extreme_baseline_wander(self) -> None:
        ecg = synthetic_ecg(noise_sd=0.01)
        t = np.arange(ecg.shape[1]) / FS
        ecg += 3.0 * np.sin(2 * np.pi * 0.15 * t)[None, :]
        res = self.gate.evaluate(ecg)
        self.assertFalse(res.passed)
        self.assertTrue(any(r.endswith("EXTREME_BASELINE_WANDER") for r in res.failure_reasons))

    def test_short_or_malformed_input(self) -> None:
        with self.assertRaises(InputError):
            self.gate.evaluate(synthetic_ecg(duration_s=2.0))
        with self.assertRaises(InputError):
            self.gate.evaluate(np.zeros((8, 10 * FS)))

    def test_load_ecg_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            npy = Path(d) / "ecg.npy"
            np.save(npy, synthetic_ecg())
            self.assertEqual(load_ecg(npy).shape, (12, 10 * FS))
            csv = Path(d) / "ecg.csv"
            np.savetxt(csv, synthetic_ecg().T, delimiter=",")
            self.assertEqual(load_ecg(csv).shape, (12, 10 * FS))
            with self.assertRaises(InputError):
                load_ecg(Path(d) / "missing.npy")


# --------------------------------------------------------------------------- #
# Orchestrator integration
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_ROUTER, "pydicom/torch not installed")
class TestOrchestratorIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.good_engine = GeometricSegmentationEngine(8000, 8000, 3000)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_chest_dicom_routes_to_tier1(self) -> None:
        path = write_dicom(self.dir / "chest.dcm", to_counts(chest_like(512), 12))
        out = execute_tier0_precheck(path, "XRAY", "cpu", thorax_engine=self.good_engine)
        self.assertTrue(out["passed"], out["errors"])
        self.assertEqual(out["route"], "TIER1_THORAX")
        self.assertEqual(tuple(out["tensor"].shape), (1, 1, 224, 224))
        self.assertEqual(out["tensor"].dtype, torch.float32)
        self.assertGreaterEqual(float(out["tensor"].min()), -1024.0 - 1e-3)
        self.assertLessEqual(float(out["tensor"].max()), 1024.0 + 1e-3)
        for stage in ("M01", "M03", "M02"):
            self.assertTrue(out["stages"][stage]["passed"])
        self.assertEqual(out["errors"], [])

    def test_abdomen_dicom_locked_out_at_m03(self) -> None:
        path = write_dicom(self.dir / "abd.dcm", to_counts(abdomen_like(512), 12), body_part="CHEST")
        out = execute_tier0_precheck(path, "XRAY", "cpu", thorax_engine=self.good_engine)
        self.assertFalse(out["passed"])
        self.assertEqual(out["halted_at"], "M03")
        self.assertEqual(out["errors"][0]["code"], "ABDOMEN_CONFIRMED")
        self.assertIsNone(out["tensor"])
        self.assertNotIn("M02", out["stages"])

    def test_thorax_hard_stop_halts_at_m02(self) -> None:
        path = write_dicom(self.dir / "tiny.dcm", to_counts(chest_like(512), 12))
        out = execute_tier0_precheck(path, "XRAY", "cpu", thorax_engine=GeometricSegmentationEngine(100, 100, 100))
        self.assertFalse(out["passed"])
        self.assertEqual(out["halted_at"], "M02")
        self.assertEqual(out["errors"][0]["code"], "THORACIC_MASS_BELOW_THRESHOLD")
        self.assertIn("Traceback", out["errors"][0]["trace"])

    def test_corrupt_dicom_halts_at_m01(self) -> None:
        bad = self.dir / "bad.dcm"
        bad.write_bytes(b"garbage" * 100)
        out = execute_tier0_precheck(bad, "XRAY", "cpu", thorax_engine=self.good_engine)
        self.assertFalse(out["passed"])
        self.assertEqual(out["halted_at"], "M01")
        self.assertFalse(out["stages"]["M01"]["passed"])

    def test_ecg_routes_to_tier2(self) -> None:
        path = self.dir / "ecg.npy"
        np.save(path, synthetic_ecg())
        out = execute_tier0_precheck(path, "ECG", "cpu")
        self.assertTrue(out["passed"], out["errors"])
        self.assertEqual(out["route"], "TIER2_BIOSIGNAL")
        self.assertEqual(tuple(out["tensor"].shape), (1, 12, 10 * FS))
        self.assertAlmostEqual(float(out["tensor"].std(dim=-1).mean()), 1.0, places=2)

    def test_flatline_and_noisy_ecg_rejected(self) -> None:
        flat, noisy = self.dir / "flat.npy", self.dir / "noisy.npy"
        np.save(flat, np.zeros((12, 10 * FS)))
        np.save(noisy, synthetic_ecg(noise_sd=1.0, seed=5))
        for p in (flat, noisy):
            out = execute_tier0_precheck(p, "ECG", "cpu")
            self.assertFalse(out["passed"])
            self.assertEqual(out["halted_at"], "M04")
            self.assertIsNone(out["tensor"])

    def test_invalid_modality_and_device(self) -> None:
        out = execute_tier0_precheck(self.dir / "x", "MRI", "cpu")
        self.assertEqual(out["errors"][0]["code"], "INVALID_MODALITY_TYPE")
        out = execute_tier0_precheck(self.dir / "x", "ECG", "not_a_device")
        self.assertEqual(out["errors"][0]["code"], "INVALID_DEVICE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
