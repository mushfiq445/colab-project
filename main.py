import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.router import validate_two_factor_organ
from src.inference import load_diagnostic_backbone, preprocess_xray, run_inference
from src.explainability import evaluate_ups_gate
from src.safety_engine import evaluate_epistemic_uncertainty, evaluate_contraindications

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM INITIALIZATION] Running on {device}")

    drive_dir = "/content/drive/MyDrive/XRay_Predictions"
    img_path = os.path.join(drive_dir, "images", "pneumonia.jpg")
    audit_file = os.path.join(drive_dir, "safety_audit.txt")
    pred_file = os.path.join(drive_dir, "predictions.txt")
    os.makedirs(drive_dir, exist_ok=True)

    simulated_lung_mask = np.ones((224, 224), dtype=np.uint8) * 255
    dicom_tag = "CHEST"

    if not validate_two_factor_organ(dicom_tag, simulated_lung_mask):
        print("[ABORT] Pipeline halted by Layer 1 Organ Router.")
        return

    if not os.path.exists(img_path):
        print(f"[ERROR] Target image missing at {img_path}.")
        return

    model = load_diagnostic_backbone(device)
    input_tensor, _ = preprocess_xray(img_path, device)
    pathologies = model.pathologies
    predictions = run_inference(model, input_tensor, pathologies)

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_pathology, top_score = sorted_preds[0]
    top_class_idx = pathologies.index(top_pathology)

    print(f"\n[DIAGNOSTIC BACKBONE] Top Finding: {top_pathology} ({top_score:.4f})")

    target_layer = model.features.denseblock4.denselayer16.conv2
    _, aamr, ups_status = evaluate_ups_gate(
        model, target_layer, input_tensor, top_class_idx,
        simulated_lung_mask, diagnostic_threshold=0.50,
        aamr_threshold=0.85, top_score=top_score
    )
    print(f"[SAFETY GATE] Layer 2 UPS -> {ups_status}")

    prob_array = np.array(list(predictions.values()))
    l3_passed, entropy, ambig_count, _ = evaluate_epistemic_uncertainty(prob_array)
    print(f"[SAFETY GATE] Layer 3 Epistemic -> Mean Entropy: {entropy:.3f} bits, Ambiguous: {ambig_count}")

    patient_vitals = {"HeartRate": 105, "SystolicBP": 86}
    patient_labs = {"eGFR": 24.5, "Creatinine": 2.3, "Platelets": 85000, "INR": 2.1, "Lactate": 4.6}
    _, interlocks = evaluate_contraindications(patient_vitals, patient_labs)
    for alert in interlocks:
        print(f"[SAFETY INTERLOCK] Layer 4 -> {alert}")

    with open(pred_file, "w") as f:
        f.write("Thoracic Diagnostic Predictions (18 Classes):\n" + "=" * 45 + "\n")
        for k, v in sorted_preds:
            f.write(f"{k:<25}: {v:.4f}\n")

    with open(audit_file, "w") as f:
        f.write("Clinical SaMD Safety Audit Report\n" + "=" * 45 + "\n")
        f.write(f"Layer 1 Organ Router Passed: True\n")
        f.write(f"Layer 2 UPS Status         : {ups_status} (AAMR: {aamr:.3f})\n")
        f.write(f"Layer 3 Epistemic Entropy  : {entropy:.3f} bits\n")
        f.write("Layer 4 Interlocks Triggered:\n")
        for alert in interlocks:
            f.write(f"  - {alert}\n")

    print(f"\n[PIPELINE COMPLETE] Outputs persisted to {drive_dir}")

if __name__ == "__main__":
    main()
