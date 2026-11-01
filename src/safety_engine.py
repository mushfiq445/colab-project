import numpy as np

def evaluate_epistemic_uncertainty(probabilities: np.ndarray, entropy_cutoff: float = 0.85, 
                                   ambiguity_range=(0.45, 0.65), max_ambiguous_classes: int = 5):
    probs = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    binary_entropy = - (probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
    mean_entropy = float(np.mean(binary_entropy))
    ambiguous_count = int(np.sum((probs >= ambiguity_range[0]) & (probs <= ambiguity_range[1])))

    rejection_reasons = []
    if mean_entropy > entropy_cutoff:
        rejection_reasons.append(f"Mean Entropy {mean_entropy:.3f} bits > {entropy_cutoff} bits")
    if ambiguous_count >= max_ambiguous_classes:
        rejection_reasons.append(f"{ambiguous_count} pathologies clustered in equivocal range {ambiguity_range}")

    passed = len(rejection_reasons) == 0
    return passed, mean_entropy, ambiguous_count, rejection_reasons

def evaluate_contraindications(vitals, labs, *, config=None, now=None, requested_actions=None):
    """Compatibility wrapper around the fail-closed interlock engine (src/interlocks.py).

    Returns ``(passed, messages)``. ``passed`` is False when any non-CLEAR rule (TRIGGERED or
    DATA_INVALID) blocks one of ``requested_actions`` (None = every action). ``messages`` lists
    every TRIGGERED / DATA_INVALID rule, including ones that do not block the requested actions.
    Thresholds come from config/clinical_interlocks.yaml; missing data never defaults to healthy.
    """
    from src.interlocks import evaluate_interlocks, load_interlock_config

    cfg = config if config is not None else load_interlock_config()
    result = evaluate_interlocks({"vitals": vitals, "labs": labs}, config=cfg, now=now)
    passed = not result.blocking_rules(requested_actions)
    return passed, result.legacy_messages()
