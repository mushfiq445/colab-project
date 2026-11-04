"""Layer 3: Epistemic Uncertainty & Entropy Normalization (Task 4/R5)."""

from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np

def evaluate_epistemic_uncertainty(
    predictions: np.ndarray,
    entropy_cutoff: float = 0.85,
    ambiguity_range: Tuple[float, float] = (0.45, 0.65),
    max_ambiguous_classes: int = 5,
) -> Tuple[bool, float, int, Dict[str, Any]]:
    preds = np.asarray(predictions, dtype=np.float64)
    if not np.all(np.isfinite(preds)):
        return False, 1.0, len(preds), {"error": "NON_FINITE_PREDICTIONS"}
    
    eps = 1e-9
    p = np.clip(preds, eps, 1.0 - eps)
    class_entropies = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    
    max_entropy = float(np.max(class_entropies))
    mean_entropy = float(np.mean(class_entropies))
    ambig_count = int(np.sum((preds >= ambiguity_range[0]) & (preds <= ambiguity_range[1])))
    
    passed = bool((max_entropy <= entropy_cutoff) and (ambig_count <= max_ambiguous_classes))
    details = {
        "max_class_entropy": max_entropy,
        "mean_class_entropy": mean_entropy,
        "ambiguous_classes_count": ambig_count,
        "failed_entropy": max_entropy > entropy_cutoff,
        "failed_ambiguity": ambig_count > max_ambiguous_classes,
    }
    return passed, max_entropy, ambig_count, details

def evaluate_contraindications(vitals=None, labs=None, **kwargs):
    """Legacy static wrapper for test suite compatibility."""
    if not vitals or not labs:
        return False, ["MISSING_PROFILE_DATA"]
    reasons = []
    if labs.get("eGFR", 100) < 30.0:
        reasons.append("L4_RENAL_LOCKOUT")
    if vitals.get("SystolicBP", 100) < 90:
        reasons.append("L4_HEMODYNAMIC_INSTABILITY")
    return len(reasons) == 0, reasons
