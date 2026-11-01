import os
import sys
import glob
import traceback
import uuid
import urllib.request
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.router import load_pspnet_segmenter, segment_thoracic_anatomy, validate_two_factor_organ
from src.inference import load_diagnostic_backbone, preprocess_xray, run_inference
from src.explainability import evaluate_ups_gate
from src.safety_engine import evaluate_epistemic_uncertainty
from src.storage_router import HybridStorageRouter
from src.tier0_thorax_router import MIN_THORACIC_PIXELS
from src.interlocks import load_interlock_config
from src.audit_log import AuditLog
from src.scan_audit import finalize_scan, record_pipeline_failure, append_manifest

# Actions this pipeline performs. Layer 4 locks only block rules that cover these actions
# (HCE-03 blocks "*"). Add "contrast_ct" / "tpa_recommendation" when those tasks exist.
PHASE1_REQUESTED_ACTIONS = ("cxr_triage",)

# Thresholds actually passed to each layer; the same dict is written into every audit record.
PHASE1_THRESHOLDS = {
    "l1_min_thoracic_pixels": MIN_THORACIC_PIXELS,
    "l2_diagnostic_threshold": 0.30,
    "l2_aamr_threshold": 0.85,
    "l3_entropy_cutoff": 0.85,
    "l3_ambiguity_range": [0.45, 0.65],
    "l3_max_ambiguous_classes": 5,
}

def ensure_seed_assets(images_dir: str):
    """Ensures a genuine benchmark seed radiograph and clinical sidecar exist."""
    os.makedirs(images_dir, exist_ok=True)
    img_path = os.path.join(images_dir, "pneumonia.jpg")
    json_path = os.path.join(images_dir, "pneumonia_vitals.json")

    needs_download = True
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as check_im:
                if check_im.size[0] >= 224 and check_im.size[1] >= 224:
                    needs_download = False
        except Exception:
            needs_download = True

    if needs_download:
        print(f"[FETCH SEED] Ingesting genuine benchmark radiograph into {img_path}...")
        extracted = False

        # Method 1: Check torchxrayvision bundled sample test data
        try:
            import torchxrayvision as xrv
            xrv_dir = os.path.dirname(xrv.__file__)
            candidate_samples = glob.glob(os.path.join(xrv_dir, "**", "*.png"), recursive=True) + \
                                glob.glob(os.path.join(xrv_dir, "**", "*.jpg"), recursive=True)
            for cand in candidate_samples:
                if "test" in cand.lower() or "sample" in cand.lower() or "chest" in cand.lower():
                    im = Image.open(cand).convert("L")
                    im.save(img_path)
                    print(f"[FETCH SEED] Bundled TorchXRayVision scan extracted from {cand}")
                    extracted = True
                    break
        except Exception as e:
            print(f"[FETCH SEED NOTE] Bundled asset extraction skipped: {e}")

        # Method 2: Download verified public domain NIH / ChestX-ray sample
        if not extracted:
            urls = [
                "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
                "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/test_xray.png"
            ]
            for url in urls:
                try:
                    print(f"[FETCH SEED] Downloading genuine radiograph from {url}...")
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as resp, open(img_path, 'wb') as out_f:
                        out_f.write(resp.read())
                    im = Image.open(img_path).convert("L")
                    im.save(img_path)
                    print(f"[FETCH SEED] Radiograph successfully saved ({im.size}).")
                    extracted = True
                    break
                except Exception as err:
                    print(f"[FETCH SEED] Download attempt failed for {url}: {err}")

        if not extracted:
            print("[ERROR] Could not extract or download a genuine chest radiograph.")

    if not os.path.exists(json_path):
        print(f"[CLINICAL DATA] No clinical profile at {json_path}. Scans using it will be HARD_LOCKOUT "
              "until a real profile with units/timestamps is supplied (no default values are generated).")

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

def _sync(router: HybridStorageRouter, path: str, failures: list):
    """Route a file and make a failed sync visible (the local copy remains authoritative)."""
    try:
        ok = router.route_file(path)
    except Exception as exc:  # e.g. rclone not installed; the record is already safely written locally
        print(f"[AUDIT WARNING] remote sync raised {type(exc).__name__}: {exc}")
        ok = False
    if not ok:
        failures.append(path)
        print(f"[AUDIT WARNING] {os.path.basename(path)} was NOT synced to remote storage (local copy only).")
    return ok


def process_single_scan(img_path: str, segmenter, backbone_model, device, output_dir: str,
                        router: HybridStorageRouter, *, interlock_cfg, audit_log: AuditLog,
                        run_id: str, sync_failures: list):
    scan_id = os.path.splitext(os.path.basename(img_path))[0]
    sidecar_json = os.path.splitext(img_path)[0] + "_vitals.json"
    T = PHASE1_THRESHOLDS
    model_id = getattr(backbone_model, "weights", None)
    print(f"\n=================== PROCESSING SCAN: {scan_id} ===================")

    def finalize(l1, l2, l3, predictions):
        return finalize_scan(
            scan_id=scan_id, img_path=img_path, run_id=run_id, profile_path=sidecar_json,
            interlock_cfg=interlock_cfg, requested_actions=PHASE1_REQUESTED_ACTIONS, thresholds=T,
            l1=l1, l2=l2, l3=l3, predictions=predictions, model_id=model_id,
            audit_log=audit_log, output_dir=output_dir,
        )

    # --- LAYER 1: Organ Router ---
    lung_mask, thoracic_pixels = segment_thoracic_anatomy(img_path, segmenter, device)
    l1_pass = validate_two_factor_organ("CHEST", thoracic_pixels,
                                        min_pixel_threshold=T["l1_min_thoracic_pixels"])
    l1 = {"pass": bool(l1_pass), "thoracic_pixels": int(thoracic_pixels)}
    print(f"[LAYER 1] Thoracic pixels detected: {thoracic_pixels} | Pass: {l1_pass}")

    if not l1_pass:
        print(f"[REJECT] Scan {scan_id} aborted by Layer 1 Organ Router.")
        row = finalize(l1, None, None, None)
        _sync(router, os.path.join(output_dir, f"{scan_id}_audit.txt"), sync_failures)
        return row

    # --- DIAGNOSTIC BACKBONE ---
    input_tensor, raw_img = preprocess_xray(img_path, device)
    pathologies = backbone_model.pathologies
    predictions = run_inference(backbone_model, input_tensor, pathologies)

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_pathology, top_score = sorted_preds[0]
    top_class_idx = pathologies.index(top_pathology)

    for k, v in sorted_preds[:5]:
        print(f"{k:<25}: {v:.4f}")

    # --- LAYER 2: UPS Gate ---
    target_layer = backbone_model.features.denseblock4.denselayer16.conv2
    attr_map, aamr, ups_status = evaluate_ups_gate(
        backbone_model, target_layer, input_tensor, top_class_idx,
        lung_mask, diagnostic_threshold=T["l2_diagnostic_threshold"],
        aamr_threshold=T["l2_aamr_threshold"], top_score=top_score
    )
    l2 = {"status": ups_status, "aamr": float(aamr)}

    if attr_map is not None:
        scratch_dir = "/tmp/samd_scratch"
        os.makedirs(scratch_dir, exist_ok=True)
        heatmap_path = os.path.join(scratch_dir, f"gradcam_{scan_id}_{top_pathology.lower()}.png")
        save_gradcam_overlay(attr_map, raw_img, heatmap_path)
        print(f"[SCRATCH DISK] Saved local Grad-CAM -> {heatmap_path}")
        _sync(router, heatmap_path, sync_failures)

    # --- LAYER 3: Epistemic Uncertainty Gate ---
    prob_array = np.array(list(predictions.values()))
    l3_passed, entropy, ambig_count, _ = evaluate_epistemic_uncertainty(
        prob_array, entropy_cutoff=T["l3_entropy_cutoff"],
        ambiguity_range=tuple(T["l3_ambiguity_range"]),
        max_ambiguous_classes=T["l3_max_ambiguous_classes"],
    )
    l3 = {"passed": bool(l3_passed), "entropy": float(entropy), "ambiguous_count": int(ambig_count)}

    # --- LAYER 4 + DISPOSITION + AUDIT RECORD (fail-closed; see src/scan_audit.py) ---
    row = finalize(l1, l2, l3, predictions)
    print(f"[DISPOSITION] {row['final_disposition']} ({row['reason_codes']})")
    _sync(router, os.path.join(output_dir, f"{scan_id}_audit.txt"), sync_failures)
    return row

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM INITIALIZATION] Dynamic Batch Runner on {device}")

    router = HybridStorageRouter()

    drive_base = "/content/drive/MyDrive/XRay_Predictions"
    output_dir = drive_base if os.path.exists(drive_base) else "audit_logs"
    images_dir = os.path.join(output_dir, "images")

    # Fail fast: an invalid interlock config must stop the run, never fall back to defaults.
    interlock_cfg = load_interlock_config()
    audit_log = AuditLog(os.path.join(output_dir, "audit_log.jsonl"))
    ok, bad_line, msg = audit_log.verify()
    if not ok and os.path.exists(audit_log.path):
        raise SystemExit(f"[AUDIT] Existing audit log failed verification at line {bad_line}: {msg}")
    run_id = str(uuid.uuid4())
    print(f"[AUDIT] run_id={run_id} interlock_config_sha256={interlock_cfg.sha256[:12]}...")

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
    sync_failures = []
    for img_p in sorted(image_paths):
        try:
            record = process_single_scan(
                img_p, segmenter, backbone, device, output_dir, router,
                interlock_cfg=interlock_cfg, audit_log=audit_log, run_id=run_id, sync_failures=sync_failures)
        except Exception as exc:  # fail closed: lock the scan out and log it; never drop it silently
            traceback.print_exc()
            print(f"[HARD_LOCKOUT] Scan {os.path.basename(img_p)} failed: {type(exc).__name__}: {exc}")
            record = record_pipeline_failure(
                scan_id=os.path.splitext(os.path.basename(img_p))[0], img_path=img_p, run_id=run_id, exc=exc,
                requested_actions=PHASE1_REQUESTED_ACTIONS, thresholds=PHASE1_THRESHOLDS,
                interlock_cfg=interlock_cfg, model_id=getattr(backbone, "weights", None),
                audit_log=audit_log, output_dir=output_dir)
        manifest_records.append(record)

    csv_path = os.path.join(output_dir, "audit_manifest.csv")
    append_manifest(csv_path, manifest_records)

    ok, bad_line, msg = audit_log.verify()
    print(f"\n[AUDIT CHAIN] {'OK' if ok else 'FAILED at line ' + str(bad_line)}: {msg}")
    print(f"[BATCH COMPLETE] Processed {len(manifest_records)} scan(s).")
    _sync(router, csv_path, sync_failures)
    _sync(router, str(audit_log.path), sync_failures)
    if sync_failures:
        print(f"[AUDIT WARNING] {len(sync_failures)} file(s) were not synced remotely; "
              "only local copies exist. Re-run backup_to_drive.py or fix rclone before relying on them.")

if __name__ == "__main__":
    main()
