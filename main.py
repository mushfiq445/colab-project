import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.router import load_pspnet_segmenter, segment_thoracic_anatomy, validate_two_factor_organ
from src.inference import load_diagnostic_backbone, preprocess_xray, run_inference
from src.explainability import evaluate_ups_gate
from src.safety_engine import evaluate_epistemic_uncertainty, evaluate_contraindications

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM INITIALIZATION] Running on {device}")

    drive_dir = "/content/drive/MyDrive/XRay_Predictions"
    img_path = os.path.join(drive_dir, "images", "pneumonia.jpg")

    if not os.path.exists(img_path) and os.path.exists("test_image.jpg"):
        img_path = "test_image.jpg"
        drive_dir = "audit_logs"

    os.makedirs(drive_dir, exist_ok=True)
    os.makedirs("audit_logs", exist_ok=True)

    pred_file = os.path.join(drive_dir, "predictions.txt")
    audit_file = os.path.join(drive_dir, "safety_audit.txt")
    local_audit = os.path.join("audit_logs", "safety_audit.txt")
    local_preds = os.path.join("audit_logs", "predictions.txt")

    if not os.path.exists(img_path):
        print(f"[ERROR] Target image missing at {img_path}.")
        return

    # --- LAYER 1: Organ Router (Native 512x512 PSPNet) ---
    print("\n--- SECTION 5.2: TWO-FACTOR ORGAN ROUTING CHECK ---")
    segmenter = load_pspnet_segmenter(device)
    lung_mask, thoracic_pixels = segment_thoracic_anatomy(img_path, segmenter, device)

    if not validate_two_factor_organ("CHEST", thoracic_pixels):
        print("[ABORT] Pipeline halted by Layer 1 Organ Router.")
        return

    # --- DIAGNOSTIC BACKBONE (DenseNet-121) ---
    print("\n--- RUNNING DIAGNOSTIC INFERENCE ---")
    model = load_diagnostic_backbone(device)
    input_tensor, _ = preprocess_xray(img_path, device)
    pathologies = model.pathologies
    predictions = run_inference(model, input_tensor, pathologies)

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_pathology, top_score = sorted_preds[0]
    top_class_idx = pathologies.index(top_pathology)

    for k, v in sorted_preds:
        print(f"{k:<25}: {v:.4f}")

    # --- LAYER 2: UPS Gate ---
    print("\n--- SECTION 5.3: EXPLAINABILITY (GRAD-CAM) & UPS GATE ---")
    target_layer = model.features.denseblock4.denselayer16.conv2
    _, aamr, ups_status = evaluate_ups_gate(
        model, target_layer, input_tensor, top_class_idx,
        lung_mask, diagnostic_threshold=0.50,
        aamr_threshold=0.85, top_score=top_score
    )
    print(f"[SAFETY GATE] Layer 2 UPS -> {ups_status}")

    # --- LAYER 3: Epistemic Uncertainty Gate ---
    print("\n--- SECTION 5.4: EPISTEMIC UNCERTAINTY & ENTROPY GATE ---")
    prob_array = np.array(list(predictions.values()))
    l3_passed, entropy, ambig_count, _ = evaluate_epistemic_uncertainty(prob_array)
    print(f"Mean Predictive Entropy: {entropy:.4f} bits | Ambiguous Findings: {ambig_count}")

    # --- LAYER 4: Hardcoded Contraindication Engine (HCE) ---
    print("\n--- SECTION 5.5: HARDCODED CONTRAINDICATION INTERLOCKS ---")
    patient_vitals = {"HeartRate": 105, "SystolicBP": 86}
    patient_labs = {"eGFR": 24.5, "Creatinine": 2.3, "Platelets": 85000, "INR": 2.1, "Lactate": 4.6}
    _, interlocks = evaluate_contraindications(patient_vitals, patient_labs)
    for alert in interlocks:
        print(f"[VETO] {alert}")

    # --- PERSISTENCE ---
    pred_content = "Thoracic Diagnostic Predictions (18 Classes):\n" + "=" * 45 + "\n"
    for k, v in sorted_preds:
        pred_content += f"{k:<25}: {v:.4f}\n"

    audit_lines = [
        "Clinical SaMD Safety Audit Report",
        "=" * 45,
        f"Layer 1 Organ Router Passed : True (Thoracic Pixels: {thoracic_pixels})",
        f"Layer 2 UPS Status          : {ups_status} (AAMR: {aamr:.3f})",
        f"Layer 3 Epistemic Entropy   : {entropy:.4f} bits (Ambiguous: {ambig_count})",
        "Layer 4 Interlocks Triggered:"
    ]
    for alert in interlocks:
        audit_lines.append(f"  * {alert}")
    audit_content = "\n".join(audit_lines) + "\n"

    for p_path in [pred_file, local_preds]:
        with open(p_path, "w") as f:
            f.write(pred_content)

    for a_path in [audit_file, local_audit]:
        with open(a_path, "w") as f:
            f.write(audit_content)

    print(f"\n[OK] Pipeline audit recorded successfully.")

if __name__ == "__main__":
    main()
