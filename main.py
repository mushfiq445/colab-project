import os, sys, glob, traceback, uuid, urllib.request
import torch, numpy as np, matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.router import execute_tier0_precheck, load_pspnet_segmenter
from src.tier0_thorax_router import resize_image
from src.inference import load_diagnostic_backbone, run_inference
from src.explainability import evaluate_ups_gate
from src.safety_engine import evaluate_epistemic_uncertainty
from src.storage_router import HybridStorageRouter
from src.interlocks import load_interlock_config
from src.audit_log import AuditLog
from src.scan_audit import finalize_scan, record_pipeline_failure, append_manifest

PHASE1_REQUESTED_ACTIONS = ("cxr_triage",)
PHASE1_THRESHOLDS = {
    "l2_diagnostic_threshold": 0.30,
    "l2_aamr_threshold": 0.85,
    "l3_entropy_cutoff": 0.85,
    "l3_ambiguity_range": [0.45, 0.65],
    "l3_max_ambiguous_classes": 5,
}

def ensure_seed_assets(images_dir: str):
    os.makedirs(images_dir, exist_ok=True)
    img_path = os.path.join(images_dir, "pneumonia.jpg")
    json_path = os.path.join(images_dir, "pneumonia_vitals.json")
    if not os.path.exists(img_path):
        print(f"[FETCH SEED] Downloading genuine radiograph...")
        req = urllib.request.Request(
            "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req) as resp, open(img_path, "wb") as out:
            out.write(resp.read())
    if not os.path.exists(json_path):
        print(f"[CLINICAL DATA] No profile at {json_path}. Will HARD_LOCKOUT.")

def save_gradcam_overlay(attr_map, raw_img, output_path):
    if attr_map is None: return
    plt.figure(figsize=(6, 6))
    plt.imshow(raw_img, cmap="gray")
    plt.imshow(attr_map, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def _sync(router: HybridStorageRouter, path: str, failures: list):
    try: ok = router.route_file(path)
    except Exception as exc: ok = False
    if not ok: failures.append(path)
    return ok

def process_single_scan(img_path: str, segmenter, backbone_model, device, output_dir: str, router, interlock_cfg, audit_log, run_id, sync_failures):
    scan_id = os.path.splitext(os.path.basename(img_path))[0]
    sidecar_json = os.path.splitext(img_path)[0] + "_vitals.json"
    print(f"\n=================== PROCESSING SCAN: {scan_id} ===================")

    # --- TIER 0 PRE-CHECKS (M01, M03, M02) ---
    t0_result = execute_tier0_precheck(img_path, "CXR", device=device, thorax_engine=segmenter)
    m02_data = t0_result["stages"].get("M02", {})
    l1 = {"pass": t0_result["passed"], "thoracic_pixels": m02_data.get("thoracic_pixels", 0)}
    
    def finalize(l2, l3, preds):
        return finalize_scan(
            scan_id=scan_id, img_path=img_path, run_id=run_id, profile_path=sidecar_json,
            interlock_cfg=interlock_cfg, requested_actions=PHASE1_REQUESTED_ACTIONS,
            thresholds=PHASE1_THRESHOLDS, l1=l1, l2=l2, l3=l3, predictions=preds,
            model_id=getattr(backbone_model, "model_id", None) or getattr(backbone_model, "weights", None),
            model_sha256=getattr(backbone_model, "model_sha256", None),
            model_registry_sha256=getattr(backbone_model, "registry_sha256", None),
            audit_log=audit_log, output_dir=output_dir,
        )

    if not t0_result["passed"]:
        print(f"[REJECT] Scan aborted at {t0_result['halted_at']}: {t0_result['errors']}")
        row = finalize(None, None, None)
        _sync(router, os.path.join(output_dir, f"{scan_id}_audit.txt"), sync_failures)
        return row
    print(f"[LAYER 1] Thoracic pixels detected: {l1['thoracic_pixels']} | Pass: {l1['pass']}")

    # --- DIAGNOSTIC BACKBONE ---
    input_tensor = t0_result["tensor"]
    raw_pil = Image.open(img_path).convert("L")
    raw_img = resize_image(np.array(raw_pil, dtype=np.float32), 224)
    lung_mask = t0_result["masks"].get("thorax_union")
    
    pathologies = backbone_model.pathologies
    predictions = run_inference(backbone_model, input_tensor, pathologies)
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_pathology, top_score = sorted_preds[0]
    
    for k, v in sorted_preds[:5]: print(f"{k:<25}: {v:.4f}")

    # --- LAYER 2: UPS Gate ---
    target_layer = backbone_model.features.denseblock4.denselayer16.conv2
    attr_map, aamr, ups_status = evaluate_ups_gate(
        backbone_model, target_layer, input_tensor, pathologies.index(top_pathology),
        pathology_name=top_pathology, organ_masks=t0_result["masks"] if "masks" in t0_result else {},
        diagnostic_threshold=PHASE1_THRESHOLDS["l2_diagnostic_threshold"],
        aamr_threshold=PHASE1_THRESHOLDS["l2_aamr_threshold"], top_score=top_score
    )
    l2 = {"status": ups_status, "aamr": float(aamr)}
    
    if attr_map is not None:
        os.makedirs("/tmp/samd_scratch", exist_ok=True)
        hm_path = f"/tmp/samd_scratch/gradcam_{scan_id}_{top_pathology.lower()}.png"
        save_gradcam_overlay(attr_map, raw_img, hm_path)
        print(f"[SCRATCH DISK] Saved local Grad-CAM -> {hm_path}")
        _sync(router, hm_path, sync_failures)

    # --- LAYER 3: Epistemic Uncertainty Gate ---
    l3_passed, entropy, ambig_count, _ = evaluate_epistemic_uncertainty(
        np.array(list(predictions.values())), entropy_cutoff=PHASE1_THRESHOLDS["l3_entropy_cutoff"],
        ambiguity_range=tuple(PHASE1_THRESHOLDS["l3_ambiguity_range"]), max_ambiguous_classes=PHASE1_THRESHOLDS["l3_max_ambiguous_classes"],
    )
    l3 = {"passed": bool(l3_passed), "entropy": float(entropy), "ambiguous_count": int(ambig_count)}

    # --- DISPOSITION & AUDIT ---
    row = finalize(l2, l3, predictions)
    print(f"[DISPOSITION] {row['final_disposition']} ({row['reason_codes']})")
    _sync(router, os.path.join(output_dir, f"{scan_id}_audit.txt"), sync_failures)
    return row

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM INITIALIZATION] Dynamic Batch Runner on {device}")
    router = HybridStorageRouter()
    out_dir = "/content/drive/MyDrive/XRay_Predictions" if os.path.exists("/content/drive/MyDrive/XRay_Predictions") else "audit_logs"
    img_dir = os.path.join(out_dir, "images")
    interlock_cfg = load_interlock_config()
    audit_log = AuditLog(os.path.join(out_dir, "audit_log.jsonl"))

    ok, bad_line, msg = audit_log.verify()
    if not ok and os.path.exists(audit_log.path): raise SystemExit(f"[AUDIT] verification failed at {bad_line}: {msg}")

    run_id = str(uuid.uuid4())
    print(f"[AUDIT] run_id={run_id} config_sha={interlock_cfg.sha256[:12]}")
    ensure_seed_assets(img_dir)
    image_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    if not image_paths: return print(f"[ERROR] No scans in {img_dir}")

    segmenter = load_pspnet_segmenter(device)
    backbone = load_diagnostic_backbone(device)
    manifest_records, sync_failures = [], []

    for img_p in sorted(image_paths):
        try:
            record = process_single_scan(img_p, segmenter, backbone, device, out_dir, router, interlock_cfg, audit_log, run_id, sync_failures)
        except Exception as exc:
            traceback.print_exc()
            print(f"[HARD_LOCKOUT] Scan {os.path.basename(img_p)} failed: {exc}")
            record = record_pipeline_failure(
                scan_id=os.path.splitext(os.path.basename(img_p))[0], img_path=img_p, run_id=run_id, exc=exc,
                requested_actions=PHASE1_REQUESTED_ACTIONS, thresholds=PHASE1_THRESHOLDS, interlock_cfg=interlock_cfg,
                model_id=getattr(backbone, "model_id", None), model_sha256=getattr(backbone, "model_sha256", None),
                model_registry_sha256=getattr(backbone, "registry_sha256", None), audit_log=audit_log, output_dir=out_dir)
        manifest_records.append(record)

    append_manifest(os.path.join(out_dir, "audit_manifest.csv"), manifest_records)
    ok, bad_line, msg = audit_log.verify()
    print(f"\n[AUDIT CHAIN] {'OK' if ok else f'FAILED at {bad_line}'}: {msg}")
    for p in [os.path.join(out_dir, "audit_manifest.csv"), str(audit_log.path)]: _sync(router, p, sync_failures)

if __name__ == "__main__": main()
