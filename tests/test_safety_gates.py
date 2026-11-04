import pytest
import numpy as np
from datetime import datetime, timezone
from src.router import validate_two_factor_organ
from src.explainability import calculate_aamr
from src.safety_engine import evaluate_epistemic_uncertainty, evaluate_contraindications

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
VITALS_TS = "2026-11-02T11:50:00Z"
LABS_TS = "2026-11-02T06:00:00Z"


# --- LAYER 1 TESTS ---
def test_layer1_valid_chest():
    assert validate_two_factor_organ("CHEST", thoracic_pixels=250000, min_pixel_threshold=100000) is True


def test_layer1_rejection_wrong_dicom_tag():
    assert validate_two_factor_organ("ABDOMEN", thoracic_pixels=150000, min_pixel_threshold=100000) is False


def test_layer1_rejection_low_pixel_count():
    assert validate_two_factor_organ("CHEST", thoracic_pixels=450, min_pixel_threshold=100000) is False


# --- LAYER 2 TESTS ---
def test_layer2_aamr_inside_mask():
    attr = np.array([[1.0, 1.0], [0.0, 0.0]])
    mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    import math
    assert math.isclose(calculate_aamr(attr, mask), 1.0, rel_tol=1e-5)


def test_layer2_aamr_outside_mask():
    attr = np.array([[0.0, 0.0], [1.0, 1.0]])
    mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    assert calculate_aamr(attr, mask) == 0.0


# --- LAYER 3 TESTS ---
def test_layer3_low_entropy_pass():
    # 18 highly confident negative predictions (0.01)
    preds = np.full(18, 0.01)
    passed, max_ent, ambig_count, _ = evaluate_epistemic_uncertainty(preds, entropy_cutoff=0.85, max_ambiguous_classes=2)
    assert passed is True
    assert max_ent < 0.1
    assert ambig_count == 0

def test_layer3_rejection_high_entropy():
    # 17 confident negatives, 1 highly uncertain (0.5) prediction
    preds = np.full(18, 0.01)
    preds[3] = 0.5 
    passed, max_ent, ambig_count, details = evaluate_epistemic_uncertainty(preds, entropy_cutoff=0.85, max_ambiguous_classes=2)
    assert passed is False
    assert max_ent > 0.85  # p=0.5 should equal 1.0 bit
    assert details["failed_entropy"] is True

def test_layer4_renal_lockout():
    vitals = {"HeartRate": 70, "SystolicBP": 120, "collected_at": VITALS_TS}
    labs = {"eGFR": 20.0, "Creatinine": 2.5, "Platelets": 200000, "INR": 1.0, "Lactate": 1.0, "collected_at": LABS_TS}
    passed, interlocks = evaluate_contraindications(vitals, labs, now=NOW)
    assert passed is False
    assert any("RENAL CONTRAST LOCKOUT" in i for i in interlocks)


def test_layer4_normal_pass():
    vitals = {"HeartRate": 72, "SystolicBP": 120, "collected_at": VITALS_TS}
    labs = {"eGFR": 95.0, "Creatinine": 0.8, "Platelets": 250000, "INR": 1.0, "Lactate": 1.1, "collected_at": LABS_TS}
    passed, interlocks = evaluate_contraindications(vitals, labs, now=NOW)
    assert passed is True
    assert len(interlocks) == 0


def test_layer4_missing_data_fails_closed():
    passed, interlocks = evaluate_contraindications({}, {}, now=NOW)
    assert passed is False
    assert any("CANNOT BE EVALUATED" in i for i in interlocks)
