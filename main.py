import os
import sys
import csv
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.router import load_pspnet_segmenter, segment_thoracic_anatomy, validate_two_factor_organ
from src.inference import load_diagnostic_backbone, preprocess_xray, run_inference
from src.explainability import evaluate_ups_gate, compute_gradcam
from src.safety_engine import evaluate_epistemic_uncertainty, evaluate_contraindications

def save_gradcam_overlay(attr_map, raw_img, output_path):
    if attr_map is None:
        return
    plt.figure(figsize=(6, 6))
    plt.imshow(raw_img, cmap="gray")
    plt.imshow(attr_map, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def process_single_scan(img_path: str, segmenter, backbone_model, device, output_dir: str):
    scan_id = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n=================== PROCESSING SCAN: {scan_id} ===================")
    
    record = {
        "scan_id": scan_id,
        "l1_organ_pass": False,
        "thoracic_pixels": 0,
        "top_finding": "None",
        "top_score": 0.0,
        "l2_ups_status": "N/A",
        "aamr": 0.0,
        "l3_entropy": 0.0,
        "l3_ambiguous_count": 0,
        "l4_vetoes": "None",
        "final_disposition": "REJECTED"
    }

    # --- LAYER 1: Organ Router ---
    lung_mask, thoracic_pixels = segment_thoracic_anatomy(img_path, segmenter, device)
    record["thoracic_pixels"] = thoracic_pixels
    l1_pass = validate_two_factor_organ("CHEST", thoracic_pixels)
    record["l1_organ_pass"] = l1_pass

    if not l1_pass:
        print(f"[REJECT] Scan {scan_id} aborted by Layer 1 Organ Router.")
        return record

    # --- DIAGNOSTIC BACKBONE ---
    input_tensor, raw_img = preprocess_xray(img_path, device)
    pathologies = backbone_model.pathologies
    predictions = run_inference(backbone_model, input_tensor, pathologies)

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_pathology, top_score = sorted_preds[0]
    top_class_idx = pathologies.index(top_pathology)
    
    record["top_finding"] = top_pathology
    record["top_score"] = round(float(top_score), 4)

    for k, v in sorted_preds[:5]:
        print(f"{k:<25}: {v:.4f}")

    # --- LAYER 2: UPS Gate & Heatmap Persistence ---
    target_layer = backbone_model.features.denseblock4.denselayer16.conv2
    attr_map, aamr, ups_status = evaluate_ups_gate(
        backbone_model, target_layer, input_tensor, top_class_idx,
        lung_mask, diagnostic_threshold=0.30,
        aamr_threshold=0.85, top_score=top_score
    )
    record["l2_ups_status"] = ups_status
    record["aamr"] = round(float(aamr), 3)

    # Save Grad-CAM image directly to Drive
    if attr_map is not None:
        heatmap_path = os.path.join(output_dir, f"gradcam_{top_pathology.lower()}.png")
        save_gradcam_overlay(attr_map, raw_img, heatmap_path)
        print(f"[SAVED] Grad-CAM heatmap -> {heatmap_path}")

    # --- LAYER 3: Epistemic Uncertainty Gate ---
    prob_array = np.array(list(predictions.values()))
    l3_passed, entropy, ambig_count, _ = evaluate_epistemic_uncertainty(prob_array)
    record["l3_entropy"] = round(float(entropy), 4)
    record["l3_ambiguous_count"] = ambig_count

    # --- LAYER 4: Clinical Contraindications ---
    patient_vitals = {"HeartRate": 105, "SystolicBP": 86}
    patient_labs = {"eGFR": 24.5, "Creatinine": 2.3, "Platelets": 85000, "INR": 2.1, "Lactate": 4.6}
    l4_passed, interlocks = evaluate_contraindications(patient_vitals, patient_labs)
    record["l4_vetoes"] = "; ".join(interlocks) if interlocks else "None"

    # --- DISPOSITION ---
    if l1_pass and ("REJECTED" not in ups_status) and l3_passed and l4_passed:
        record["final_disposition"] = "CLEARED_FOR_REVIEW"
    elif not l4_passed:
        record["final_disposition"] = "HELD_BY_CONTRAINDICATION"
    else:
        record["final_disposition"] = "FLAGGED_FOR_AUDIT"

    # Save detailed scan report to Drive
    scan_report = os.path.join(output_dir, f"{scan_id}_audit.txt")
    with open(scan_report, "w") as f:
        f.write(f"SaMD Clinical Safety Audit: {scan_id}\n" + "=" * 45 + "\n")
        f.write(f"Top Finding: {top_pathology} ({top_score:.4f})\n")
        f.write(f"Layer 1 Organ Pass: {l1_pass} ({thoracic_pixels} px)\n")
        f.write(f"Layer 2 UPS Status: {ups_status} (AAMR: {aamr:.3f})\n")
        f.write(f"Layer 3 Entropy: {entropy:.4f} bits (Ambiguous: {ambig_count})\n")
        f.write("Layer 4 Interlocks:\n")
        for i in interlocks:
            f.write(f"  * {i}\n")

    return record

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM INITIALIZATION] Batch Runner on {device}")

    # Primary output target is Google Drive
    drive_base = "/content/drive/MyDrive/XRay_Predictions"
    output_dir = drive_base if os.path.exists(drive_base) else "audit_logs"
    images_dir = os.path.join(output_dir, "images")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    if not image_paths:
        print(f"[ERROR] No scan images found in {images_dir}")
        return

    print(f"Found {len(image_paths)} scan(s) in {images_dir}")

    segmenter = load_pspnet_segmenter(device)
    backbone = load_diagnostic_backbone(device)

    manifest_records = []
    for img_p in sorted(image_paths):
        record = process_single_scan(img_p, segmenter, backbone, device, output_dir)
        manifest_records.append(record)

    # Save audit_manifest.csv directly into Google Drive
    csv_path = os.path.join(output_dir, "audit_manifest.csv")
    fields = [
        "scan_id", "l1_organ_pass", "thoracic_pixels", "top_finding",
        "top_score", "l2_ups_status", "aamr", "l3_entropy",
        "l3_ambiguous_count", "l4_vetoes", "final_disposition"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_records)

    print(f"\n[BATCH COMPLETE] Processed {len(manifest_records)} scan(s).")
    print(f"[DRIVE PERSISTED] Manifest saved -> {csv_path}")

if __name__ == "__main__":
    main()
