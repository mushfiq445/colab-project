import os
import sys
import csv
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.router import load_pspnet_segmenter, segment_thoracic_anatomy, validate_two_factor_organ
from src.inference import load_diagnostic_backbone, preprocess_xray, run_inference
from src.explainability import evaluate_ups_gate
from src.safety_engine import evaluate_epistemic_uncertainty, evaluate_contraindications
from src.storage_router import HybridStorageRouter

def ensure_seed_assets(images_dir: str):
    """Ensures a genuine benchmark seed scan and clinical sidecar exist using XRV sample dataset."""
    os.makedirs(images_dir, exist_ok=True)
    img_path = os.path.join(images_dir, "pneumonia.jpg")
    json_path = os.path.join(images_dir, "pneumonia_vitals.json")

    if not os.path.exists(img_path):
        print(f"[FETCH SEED] Extracting genuine benchmark radiograph into {img_path}...")
        try:
            import torchxrayvision as xrv
            # Ingests a genuine sample radiograph provided directly by torchxrayvision
            d_nih = xrv.datasets.COVID19_Dataset(views=["PA"], transform=None)
            sample_img = d_nih[0]["img"]  # Shape: (1, H, W)
            # Normalize to 0-255 uint8 grayscale image
            norm_img = sample_img[0]
            norm_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min() + 1e-8)
            norm_img = (norm_img * 255.0).astype(np.uint8)
            im = Image.fromarray(norm_img)
            im.save(img_path)
            print("[FETCH SEED] Genuine radiograph saved successfully.")
        except Exception as e:
            print(f"[FETCH SEED FALLBACK] XRV dataset fetch failed ({e}). Generating high-frequency anatomical test scan...")
            # Fallback high-resolution anatomical chest-like pattern
            n = 512
            yy, xx = np.mgrid[0:n, 0:n] / float(n)
            img = np.full((n, n), 0.62)
            for cx in (0.34, 0.66):
                img[((xx - cx) / 0.13) ** 2 + ((yy - 0.38) / 0.25) ** 2 < 1.0] = 0.15
            img[yy > 0.75] = 0.72
            img = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(img_path)

    if not os.path.exists(json_path):
        print(f"[SIDECAR INIT] Generating default clinical vitals sidecar -> {json_path}")
        default_sidecar = {
            "vitals": {"HeartRate": 78, "SystolicBP": 118},
            "labs": {"eGFR": 88.0, "Creatinine": 0.95, "Platelets": 240000, "INR": 1.05, "Lactate": 1.2}
        }
        with open(json_path, "w") as f:
            json.dump(default_sidecar, f, indent=2)

def load_patient_clinical_profile(json_path: str):
    default_vitals = {"HeartRate": 75, "SystolicBP": 120}
    default_labs = {"eGFR": 90.0, "Creatinine": 0.9, "Platelets": 250000, "INR": 1.0, "Lactate": 1.1}

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            vitals = data.get("vitals", default_vitals)
            labs = data.get("labs", default_labs)
            print(f"[SIDECAR LOADED] Clinical profile ingested from {os.path.basename(json_path)}")
            return vitals, labs
        except Exception as e:
            print(f"[WARNING] Failed to parse {json_path}: {e}. Falling back to baseline.")

    print("[DEFAULT PROFILE] Using baseline standard vitals/labs.")
    return default_vitals, default_labs

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

def process_single_scan(img_path: str, segmenter, backbone_model, device, output_dir: str, router: HybridStorageRouter):
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

    # --- LAYER 2: UPS Gate ---
    target_layer = backbone_model.features.denseblock4.denselayer16.conv2
    attr_map, aamr, ups_status = evaluate_ups_gate(
        backbone_model, target_layer, input_tensor, top_class_idx,
        lung_mask, diagnostic_threshold=0.30,
        aamr_threshold=0.85, top_score=top_score
    )
    record["l2_ups_status"] = ups_status
    record["aamr"] = round(float(aamr), 3)

    if attr_map is not None:
        scratch_dir = "/tmp/samd_scratch"
        os.makedirs(scratch_dir, exist_ok=True)
        heatmap_path = os.path.join(scratch_dir, f"gradcam_{scan_id}_{top_pathology.lower()}.png")
        save_gradcam_overlay(attr_map, raw_img, heatmap_path)
        print(f"[SCRATCH DISK] Saved local Grad-CAM -> {heatmap_path}")
        router.route_file(heatmap_path)

    # --- LAYER 3: Epistemic Uncertainty Gate ---
    prob_array = np.array(list(predictions.values()))
    l3_passed, entropy, ambig_count, _ = evaluate_epistemic_uncertainty(prob_array)
    record["l3_entropy"] = round(float(entropy), 4)
    record["l3_ambiguous_count"] = ambig_count

    # --- LAYER 4: Dynamic Sidecar Vitals & Contraindications ---
    sidecar_json = os.path.splitext(img_path)[0] + "_vitals.json"
    vitals, labs = load_patient_clinical_profile(sidecar_json)
    l4_passed, interlocks = evaluate_contraindications(vitals, labs)
    record["l4_vetoes"] = "; ".join(interlocks) if interlocks else "None"

    # --- DISPOSITION ---
    if l1_pass and ("REJECTED" not in ups_status) and l3_passed and l4_passed:
        record["final_disposition"] = "CLEARED_FOR_REVIEW"
    elif not l4_passed:
        record["final_disposition"] = "HELD_BY_CONTRAINDICATION"
    else:
        record["final_disposition"] = "FLAGGED_FOR_AUDIT"

    # Save scan-level text report and route it
    scan_report = os.path.join(output_dir, f"{scan_id}_audit.txt")
    with open(scan_report, "w") as f:
        f.write(f"SaMD Clinical Safety Audit: {scan_id}\n" + "=" * 45 + "\n")
        f.write(f"Disposition: {record['final_disposition']}\n")
        f.write(f"Top Finding: {top_pathology} ({top_score:.4f})\n")
        f.write(f"Layer 1 Organ Pass: {l1_pass} ({thoracic_pixels} px)\n")
        f.write(f"Layer 2 UPS Status: {ups_status} (AAMR: {aamr:.3f})\n")
        f.write(f"Layer 3 Entropy: {entropy:.4f} bits (Ambiguous: {ambig_count})\n")
        f.write("Layer 4 Interlocks:\n")
        for i in interlocks:
            f.write(f"  * {i}\n")

    router.route_file(scan_report)
    return record

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM INITIALIZATION] Dynamic Batch Runner on {device}")

    router = HybridStorageRouter()

    drive_base = "/content/drive/MyDrive/XRay_Predictions"
    output_dir = drive_base if os.path.exists(drive_base) else "audit_logs"
    images_dir = os.path.join(output_dir, "images")

    ensure_seed_assets(images_dir)

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
        record = process_single_scan(img_p, segmenter, backbone, device, output_dir, router)
        manifest_records.append(record)

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
    router.route_file(csv_path)

if __name__ == "__main__":
    main()
