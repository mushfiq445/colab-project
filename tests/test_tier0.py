"""Tier 0 Pre-Check Suite (M01 DICOM, M02 Thorax, M03 Abdomen, M04 ECG, Orchestrator)."""

from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

from src.tier0_abdomen_router import AbdomenRouter
from src.tier0_biosignal_gate import (
    BiosignalGate,
    EcgInput,
    bandpass_filter_ecg,
    load_ecg,
    zscore_leads,
)
from src.tier0_dicom_validator import DicomValidator, LOSSY_TRANSFER_SYNTAXES
from src.tier0_errors import (
    BiosignalQualityError,
    DicomValidationError,
    InputError,
    OrganRoutingError,
    Tier0Error,
)
from src.tier0_thorax_router import (
    MIN_THORACIC_PIXELS,
    SEGMENTATION_SIZE,
    ThoraxRouter,
    resize_image,
    to_xrv_range,
)
from src.router import execute_tier0_precheck, validate_two_factor_organ


class FakeSegmentationEngine:
    def __init__(self, lung_pixels: int = 150000) -> None:
        self.lung_pixels = int(lung_pixels)

    def predict_masks(self, xrv_image: np.ndarray):
        mask = np.zeros((SEGMENTATION_SIZE, SEGMENTATION_SIZE), dtype=bool)
        flat = mask.ravel()
        flat[: min(self.lung_pixels, flat.size)] = True
        return {"Left Lung": mask, "Right Lung": np.zeros_like(mask), "Heart": np.zeros_like(mask)}


def make_mock_dicom(**overrides):
    rng = np.random.default_rng(42)
    pixels = rng.integers(100, 2000, size=(128, 128), dtype=np.uint16)
    attrs = {
        "file_meta": SimpleNamespace(TransferSyntaxUID="1.2.840.10008.1.2.1"),
        "LossyImageCompression": "00",
        "BurnedInAnnotation": "NO",
        "Modality": "DX",
        "BodyPartExamined": "CHEST",
        "ViewPosition": "PA",
        "BitsStored": 12,
        "PhotometricInterpretation": "MONOCHROME2",
        "pixel_array": pixels,
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def make_clean_ecg(fs: int = 500, seconds: float = 10.0) -> EcgInput:
    t = np.linspace(0, seconds, int(fs * seconds), endpoint=False, dtype=np.float32)
    # 1.2 Hz + 8 Hz physiological-band sinusoid inside [0.5, 40] Hz passband
    lead = 1.0 * np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 8.0 * t)
    sig = np.tile(lead, (12, 1)).astype(np.float32)
    return EcgInput(signal_mv=sig, sampling_rate_hz=fs)


class TestM01DicomValidator(unittest.TestCase):
    def test_01_valid_dicom_passes(self):
        res = DicomValidator().validate_dataset(make_mock_dicom())
        self.assertTrue(res.report.hipaa_safe_harbor_compliant)
        self.assertEqual(res.report.modality, "DX")
        self.assertFalse(res.report.inverted_monochrome1)

    def test_02_missing_file_raises(self):
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate("/nonexistent/scan.dcm")
        self.assertEqual(ctx.exception.code, "M01_FILE_NOT_FOUND")

    def test_03_lossy_transfer_syntax_rejected(self):
        lossy_uid = next(iter(LOSSY_TRANSFER_SYNTAXES))
        ds = make_mock_dicom(file_meta=SimpleNamespace(TransferSyntaxUID=lossy_uid))
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_LOSSY_TRANSFER_SYNTAX")

    def test_04_lossy_compression_flag_rejected(self):
        ds = make_mock_dicom(LossyImageCompression="01")
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_LOSSY_COMPRESSION_FLAG")

    def test_05_burned_in_annotation_rejected(self):
        ds = make_mock_dicom(BurnedInAnnotation="YES")
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_BURNED_IN_ANNOTATION")

    def test_06_phi_patient_name_rejected_in_strict_mode(self):
        ds = make_mock_dicom(PatientName="DOE^JOHN")
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator(strict_phi=True).validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_PHI_DETECTED")

    def test_07_phi_anonymized_placeholder_allowed(self):
        ds = make_mock_dicom(PatientName="ANONYMIZED", PatientID="REDACTED")
        res = DicomValidator(strict_phi=True).validate_dataset(ds)
        self.assertTrue(res.report.hipaa_safe_harbor_compliant)

    def test_08_phi_non_strict_records_warning(self):
        ds = make_mock_dicom(InstitutionName="GENERAL HOSPITAL")
        res = DicomValidator(strict_phi=False).validate_dataset(ds)
        self.assertFalse(res.report.hipaa_safe_harbor_compliant)
        self.assertTrue(any("PHI_TAGS_PRESENT" in w for w in res.report.warnings))

    def test_09_unsupported_modality_rejected(self):
        ds = make_mock_dicom(Modality="OT")
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_UNSUPPORTED_MODALITY")

    def test_10_missing_body_part_rejected(self):
        ds = make_mock_dicom(BodyPartExamined="")
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_MISSING_BODY_PART")

    def test_11_insufficient_bits_stored_rejected(self):
        ds = make_mock_dicom(BitsStored=4)
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_INSUFFICIENT_BIT_DEPTH")

    def test_12_monochrome1_is_inverted(self):
        arr = np.array([[10.0, 50.0], [100.0, 200.0]], dtype=np.float32)
        ds = make_mock_dicom(PhotometricInterpretation="MONOCHROME1", pixel_array=arr)
        res = DicomValidator().validate_dataset(ds)
        self.assertTrue(res.report.inverted_monochrome1)
        np.testing.assert_allclose(res.pixel_array, 200.0 - arr)

    def test_13_unsupported_photometric_rejected(self):
        ds = make_mock_dicom(PhotometricInterpretation="RGB")
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_UNSUPPORTED_PHOTOMETRIC")

    def test_14_blank_constant_image_rejected(self):
        ds = make_mock_dicom(pixel_array=np.ones((64, 64), dtype=np.float32) * 5.0)
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_BLANK_IMAGE")

    def test_15_nan_pixels_rejected(self):
        arr = np.ones((64, 64), dtype=np.float32)
        arr[0, 0] = np.nan
        ds = make_mock_dicom(pixel_array=arr)
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_NON_FINITE_PIXELS")

    def test_16_3d_pixel_array_rejected_in_2d_validator(self):
        ds = make_mock_dicom(pixel_array=np.ones((4, 64, 64), dtype=np.float32))
        with self.assertRaises(DicomValidationError) as ctx:
            DicomValidator().validate_dataset(ds)
        self.assertEqual(ctx.exception.code, "M01_INVALID_DIMENSIONS")


class TestM02ThoraxRouter(unittest.TestCase):
    def setUp(self):
        self.img = np.linspace(0.0, 255.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    def test_17_valid_chest_passes(self):
        router = ThoraxRouter(FakeSegmentationEngine(lung_pixels=150000))
        rep = router.audit(self.img, "CHEST")
        self.assertTrue(rep.passed)
        self.assertEqual(rep.thoracic_pixels, 80000)

    def test_18_wrong_body_part_rejected(self):
        router = ThoraxRouter(FakeSegmentationEngine(lung_pixels=150000))
        with self.assertRaises(OrganRoutingError) as ctx:
            router.audit(self.img, "KNEE")
        self.assertEqual(ctx.exception.code, "M02_BODY_PART_MISMATCH")

    def test_19_low_thoracic_pixels_rejected(self):
        router = ThoraxRouter(FakeSegmentationEngine(lung_pixels=500), min_thoracic_pixels=1000)
        with self.assertRaises(OrganRoutingError) as ctx:
            router.audit(self.img, "CHEST")
        self.assertEqual(ctx.exception.code, "M02_INSUFFICIENT_THORACIC_PIXELS")

    def test_20_near_full_grid_mask_emits_warning(self):
        router = ThoraxRouter(FakeSegmentationEngine(lung_pixels=260113))
        rep = router.audit(self.img, "THORAX")
        self.assertIn("M02_MASK_NEAR_FULL_GRID_SATURATION", rep.warnings)

    def test_21_to_xrv_range_bounds(self):
        out = to_xrv_range(self.img)
        self.assertAlmostEqual(float(np.min(out)), -1024.0, places=3)
        self.assertAlmostEqual(float(np.max(out)), 1024.0, places=3)

    def test_22_to_xrv_range_rejects_flat_image(self):
        with self.assertRaises(OrganRoutingError) as ctx:
            to_xrv_range(np.zeros((32, 32), dtype=np.float32))
        self.assertEqual(ctx.exception.code, "M02_ZERO_VARIANCE_IMAGE")

    def test_23_resize_image_target_shape(self):
        out = resize_image(self.img, 224)
        self.assertEqual(out.shape, (224, 224))

    def test_24_validate_two_factor_wrapper(self):
        self.assertTrue(validate_two_factor_organ("CHEST", 1000, 1000))
        self.assertFalse(validate_two_factor_organ("CHEST", 999, 1000))
        self.assertFalse(validate_two_factor_organ("SKULL", 50000, 1000))


class TestM03AbdomenRouter(unittest.TestCase):
    def test_25_balanced_thorax_image_passes(self):
        rng = np.random.default_rng(7)
        img = rng.uniform(10.0, 200.0, size=(64, 64)).astype(np.float32)
        rep = AbdomenRouter().enforce(img)
        self.assertTrue(rep.heuristic_uncalibrated)
        self.assertLessEqual(rep.subdiaphragmatic_intensity_ratio, 2.5)

    def test_26_abdomen_heavy_scan_rejected(self):
        img = np.ones((64, 64), dtype=np.float32)
        img[:32, :] = 1.0
        img[32:, :] = 100.0
        with self.assertRaises(OrganRoutingError) as ctx:
            AbdomenRouter(max_ratio=2.5).enforce(img)
        self.assertEqual(ctx.exception.code, "M03_ABDOMEN_DOMINANT_SCAN")

    def test_27_flat_image_rejected(self):
        with self.assertRaises(OrganRoutingError) as ctx:
            AbdomenRouter().enforce(np.ones((32, 32), dtype=np.float32))
        self.assertEqual(ctx.exception.code, "M03_FLAT_IMAGE")

    def test_28_non_finite_image_rejected(self):
        img = np.ones((32, 32), dtype=np.float32)
        img[0, 0] = np.inf
        with self.assertRaises(OrganRoutingError) as ctx:
            AbdomenRouter().enforce(img)
        self.assertEqual(ctx.exception.code, "M03_NON_FINITE_IMAGE")


class TestM04BiosignalGate(unittest.TestCase):
    def test_29_clean_12lead_ecg_passes(self):
        ecg = make_clean_ecg()
        rep = BiosignalGate().enforce(ecg)
        self.assertEqual(rep.n_leads, 12)
        self.assertEqual(rep.n_samples, 5000)
        self.assertGreater(rep.snr_db, 5.0)
        self.assertIsNotNone(rep.filtered_mv)

    def test_30_wrong_sampling_rate_rejected(self):
        ecg = make_clean_ecg(fs=250)
        with self.assertRaises(BiosignalQualityError) as ctx:
            BiosignalGate(expected_fs=500).enforce(ecg)
        self.assertEqual(ctx.exception.code, "M04_SAMPLING_RATE_MISMATCH")

    def test_31_wrong_lead_count_rejected(self):
        sig = np.ones((8, 5000), dtype=np.float32)
        with self.assertRaises(BiosignalQualityError) as ctx:
            BiosignalGate().enforce(EcgInput(sig, 500))
        self.assertEqual(ctx.exception.code, "M04_INVALID_LEAD_COUNT")

    def test_32_short_duration_rejected(self):
        ecg = make_clean_ecg(seconds=2.0)
        with self.assertRaises(BiosignalQualityError) as ctx:
            BiosignalGate(min_duration_s=5.0).enforce(ecg)
        self.assertEqual(ctx.exception.code, "M04_DURATION_TOO_SHORT")

    def test_33_flatline_lead_off_rejected_with_lead_details(self):
        ecg = make_clean_ecg()
        ecg.signal_mv[3, :] = 0.0
        with self.assertRaises(BiosignalQualityError) as ctx:
            BiosignalGate().enforce(ecg)
        self.assertEqual(ctx.exception.code, "M04_LEAD_OFF_DETECTED")
        self.assertIn(3, ctx.exception.details["leads"])

    def test_34_amplitude_saturation_rejected(self):
        ecg = make_clean_ecg()
        ecg.signal_mv[0, 100] = 25.0
        with self.assertRaises(BiosignalQualityError) as ctx:
            BiosignalGate().enforce(ecg)
        self.assertEqual(ctx.exception.code, "M04_AMPLITUDE_RAIL_SATURATION")

    def test_35_high_frequency_noise_low_snr_rejected(self):
        t = np.linspace(0, 10.0, 5000, endpoint=False, dtype=np.float32)
        # 120 Hz powerline harmonic outside 0.5-40 Hz passband -> low SNR
        sig = np.tile(np.sin(2 * np.pi * 120.0 * t), (12, 1)).astype(np.float32)
        with self.assertRaises(BiosignalQualityError) as ctx:
            BiosignalGate(min_snr_db=5.0).enforce(EcgInput(sig, 500))
        self.assertEqual(ctx.exception.code, "M04_LOW_SNR")

    def test_36_zscore_leads_normalizes_per_lead(self):
        ecg = make_clean_ecg()
        z = zscore_leads(ecg.signal_mv)
        np.testing.assert_allclose(np.mean(z, axis=-1), np.zeros(12), atol=1e-5)
        np.testing.assert_allclose(np.std(z, axis=-1), np.ones(12), atol=1e-4)

    def test_37_load_ecg_from_npy_and_json(self):
        ecg = make_clean_ecg(seconds=5.0)
        with tempfile.TemporaryDirectory() as d:
            npy_p = Path(d) / "sample.npy"
            np.save(str(npy_p), ecg.signal_mv)
            loaded = load_ecg(npy_p)
            self.assertEqual(loaded.signal_mv.shape, (12, 2500))

            json_p = Path(d) / "sample.json"
            json_p.write_text(json.dumps({"sampling_rate_hz": 500, "signal_mv": ecg.signal_mv.tolist()}))
            loaded_j = load_ecg(json_p)
            self.assertEqual(loaded_j.sampling_rate_hz, 500)


class TestOrchestratorPrecheck(unittest.TestCase):
    def test_38_unknown_modality_fails_closed(self):
        res = execute_tier0_precheck("dummy.png", "PET_SCAN")
        self.assertFalse(res["passed"])
        self.assertEqual(res["halted_at"], "ORCHESTRATOR")
        self.assertEqual(res["errors"][0]["code"], "INVALID_MODALITY_TYPE")

    def test_39_standard_image_pipeline_with_fake_engine(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cxr.png"
            rng = np.random.default_rng(11)
            arr = rng.integers(20, 220, size=(128, 128), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(p)
            res = execute_tier0_precheck(
                p, "CXR", device="cpu", thorax_engine=FakeSegmentationEngine(lung_pixels=50000)
            )
            self.assertTrue(res["passed"])
            self.assertEqual(res["route"], "TIER1_THORAX")
            self.assertEqual(tuple(res["tensor"].shape), (1, 1, 224, 224))

    def test_40_biosignal_orchestrator_route(self):
        ecg = make_clean_ecg(seconds=6.0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ecg.npy"
            np.save(str(p), ecg.signal_mv)
            res = execute_tier0_precheck(p, "12LEAD_ECG", device="cpu")
            self.assertTrue(res["passed"])
            self.assertEqual(res["route"], "TIER2_BIOSIGNAL")
            self.assertEqual(tuple(res["tensor"].shape), (1, 12, 3000))


if __name__ == "__main__":
    unittest.main()
