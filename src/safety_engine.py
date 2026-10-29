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

def evaluate_contraindications(vitals: dict, labs: dict):
    interlocks = []
    egfr = labs.get("eGFR", 100)
    creatinine = labs.get("Creatinine", 0.9)
    if egfr < 30 or creatinine > 2.0:
        interlocks.append(f"[HCE-01] RENAL CONTRAST LOCKOUT: eGFR: {egfr} mL/min, Cr: {creatinine} mg/dL")

    platelets = labs.get("Platelets", 250000)
    inr = labs.get("INR", 1.0)
    if platelets < 100000 or inr > 1.7:
        interlocks.append(f"[HCE-02] COAGULOPATHY THROMBOLYSIS LOCKOUT: Platelets: {platelets}/uL, INR: {inr}")

    hr = vitals.get("HeartRate", 75)
    sbp = vitals.get("SystolicBP", 120)
    shock_index = hr / sbp if sbp > 0 else 0
    lactate = labs.get("Lactate", 1.0)
    if shock_index > 0.9 or lactate > 2.2:
        interlocks.append(f"[HCE-03] HEMODYNAMIC INSTABILITY TRIAGE LOCKOUT: SBP: {sbp} mmHg, Shock Index: {shock_index:.2f}, Lactate: {lactate} mmol/L")

    passed = len(interlocks) == 0
    return passed, interlocks
