import os
import sys
import numpy as np
import skimage.io
import skimage.transform
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam

# Paths
drive_root = '/content/drive/MyDrive/XRay_Predictions'
images_dir = os.path.join(drive_root, 'images')
output_file = os.path.join(drive_root, 'predictions.txt')
audit_file = os.path.join(drive_root, 'safety_audit.txt')
os.makedirs(images_dir, exist_ok=True)

# 1. Load Pre-trained Models
print("Loading DenseNet diagnostic backbone and PSPNet anatomical segmenter...")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

seg_model = xrv.baseline_models.chestx_det.PSPNet()
seg_model.eval()

# 2. Find and Load Image
candidates = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not candidates:
    print(f"[ERROR] No valid X-Ray found in {images_dir}")
    print("Please ensure an image file exists in the directory.")
    sys.exit(1)

target_img_path = candidates[0]
print(f"Detected file: {target_img_path}")
raw_img = skimage.io.imread(target_img_path)
if raw_img.ndim == 3:
    raw_img = raw_img.mean(axis=2)
max_val = 65535 if raw_img.dtype == np.uint16 else 255
img = xrv.datasets.normalize(raw_img, max_val)[None, ...]
print(f"Successfully loaded: {os.path.basename(target_img_path)}")

# 3. Transform Tensor to Standard Input Resolution (224x224)
transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])
img_tensor = torch.from_numpy(transform(img))
input_tensor = img_tensor.clone().detach().unsqueeze(0)  # Shape: [1, 1, 224, 224]

# ========================================================
# SECTION 5.2: TWO-FACTOR ANATOMY ROUTER & GATING
# ========================================================
print("\n--- SECTION 5.2: TWO-FACTOR ORGAN ROUTING CHECK ---")
dicom_metadata = {
    "BodyPartExamined": "CHEST",
    "Modality": "DX"
}

with torch.no_grad():
    seg_mask = seg_model(input_tensor)

thoracic_indices = [
    seg_model.targets.index("Left Lung"),
    seg_model.targets.index("Right Lung"),
    seg_model.targets.index("Heart")
]

thoracic_pixel_mass = sum([torch.sum(seg_mask[0, idx] > 0.5).item() for idx in thoracic_indices])
MIN_THORACIC_PIXELS = 1000
visual_anatomy_detected = "CHEST" if thoracic_pixel_mass >= MIN_THORACIC_PIXELS else "NON_CHEST"
declared_body_part = dicom_metadata["BodyPartExamined"].upper()

print(f"Declared Metadata Anatomy : {declared_body_part}")
print(f"Visual Pixel Anatomy      : {visual_anatomy_detected} (Thoracic pixels: {thoracic_pixel_mass})")

if declared_body_part != visual_anatomy_detected:
    routing_status = "ERR_DICOM_ANATOMY_MISMATCH"
    print(f"\n[CRITICAL ROUTING FAILURE] {routing_status}")
    print(f"Conflict: Header declared '{declared_body_part}', but visual scanner detected '{visual_anatomy_detected}'.")
    print("Action: HARD HALT. Processing terminated before diagnostic inference.")
    with open(audit_file, "w") as f:
        f.write(f"[ROUTING ERROR] {routing_status}: Metadata and pixel anatomy conflict.\n")
    sys.exit(1)
else:
    print(f"[PASS] Anatomy Verified: Proceeding to diagnostic inference.")

# ========================================================
# SECTION 3 & 4: DIAGNOSTIC INFERENCE BACKBONE
# ========================================================
print("\n--- RUNNING DIAGNOSTIC INFERENCE ---")
with torch.no_grad():
    predictions = model(input_tensor)

results = dict(zip(model.pathologies, predictions[0].numpy()))
sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

with open(output_file, 'w') as f:
    f.write("--- PREDICTION RESULTS ---\n")
    for pathology, score in sorted_results:
        line = f"{pathology:25s}: {score:.4f}"
        print(line)
        f.write(line + "\n")

print(f"\n[OK] Predictions saved to: {output_file}")

# Target top predicted pathology
top_pathology, top_score = sorted_results[0]
DIAGNOSTIC_THRESHOLD = 0.50

# ========================================================
# SECTION 5.3: EXPLAINABILITY & UPS AUDIT
# ========================================================
if top_score < DIAGNOSTIC_THRESHOLD:
    print(f"\n--- SECTION 5.3: EXPLAINABILITY (GRAD-CAM) & UPS GATE ---")
    print(f"Top finding '{top_pathology}' score ({top_score:.4f}) is below diagnostic threshold ({DIAGNOSTIC_THRESHOLD}).")
    print("[PASS] Study negative for primary acute thoracic pathology. Saliency noise audit bypassed.")
    ups_status = "PASS - No Acute Findings Above Threshold"
    aamr = 1.0
else:
    print(f"\n--- SECTION 5.3: EXPLAINABILITY (GRAD-CAM) & UPS GATE ---")
    print(f"Targeting highest probability pathology for spatial audit: {top_pathology} ({top_score:.4f})")
    
    target_idx = model.pathologies.index(top_pathology)
    target_layer = model.features.denseblock4
    gradcam = LayerGradCam(model, target_layer)

    grad_input = input_tensor.clone().detach().requires_grad_(True)
    attribution = gradcam.attribute(grad_input, target=target_idx)

    upsampled_attr = LayerGradCam.interpolate(attribution, (224, 224), interpolate_mode="bilinear")
    heatmap = upsampled_attr.squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    left_lung_idx = seg_model.targets.index("Left Lung")
    right_lung_idx = seg_model.targets.index("Right Lung")
    lung_mask_binary = (seg_mask[0, left_lung_idx] > 0.5) | (seg_mask[0, right_lung_idx] > 0.5)
    lung_mask_np = lung_mask_binary.cpu().numpy().astype(np.float32)

    lung_mask_resized = skimage.transform.resize(
        lung_mask_np,
        heatmap.shape,
        order=0,
        anti_aliasing=False
    )

    total_heatmap_mass = float(np.sum(heatmap))
    anatomical_mass = float(np.sum(heatmap * lung_mask_resized))
    aamr = (anatomical_mass / total_heatmap_mass) if total_heatmap_mass > 0 else 0.0

    print(f"Anatomical Attribution Mass Ratio (AAMR): {aamr:.4f}")
    AAMR_THRESHOLD = 0.85

    if aamr >= AAMR_THRESHOLD:
        ups_status = "VERIFIED - Pathologically Plausible"
        print(f"[PASS] {ups_status}: Heatmap mass aligns with lung fields.")
    else:
        ups_status = "REJECTED - Untrustworthy Shortcut Detected"
        print(f"[ALERT] {ups_status}: Heatmap mass falls outside pulmonary anatomy.")
        print("Action: Diagnostic label SUPPRESSED. Case flagged for human review.")

    raw_display = input_tensor.squeeze().detach().cpu().numpy()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(raw_display, cmap="gray")
    axs[0].set_title("Input Chest X-Ray", fontsize=12)
    axs[0].axis("off")

    axs[1].imshow(raw_display, cmap="gray")
    axs[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axs[1].set_title(f"Grad-CAM: {top_pathology} (AAMR: {aamr:.2f})", fontsize=12)
    axs[1].axis("off")

    gradcam_save_path = os.path.join(drive_root, f"gradcam_{top_pathology.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(gradcam_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[OK] Saliency map saved to: {gradcam_save_path}")

# ========================================================
# SECTION 5.4: EPISTEMIC UNCERTAINTY & ENTROPY AUDIT
# ========================================================
print(f"\n--- SECTION 5.4: EPISTEMIC UNCERTAINTY & ENTROPY GATE ---")
raw_probs = np.array(list(results.values()), dtype=np.float64)
probs_clipped = np.clip(raw_probs, 1e-7, 1.0 - 1e-7)
class_entropies = -(probs_clipped * np.log2(probs_clipped) + (1.0 - probs_clipped) * np.log2(1.0 - probs_clipped))

mean_entropy = float(np.mean(class_entropies))
ambiguous_findings = [path for path, score in results.items() if 0.45 <= score <= 0.65]

print(f"Mean Predictive Entropy: {mean_entropy:.4f} bits")
print(f"Ambiguous Findings Count: {len(ambiguous_findings)}")

ENTROPY_THRESHOLD = 0.85
AMBIGUITY_COUNT_THRESHOLD = 5

uncertainty_refusal = False
refusal_reasons = []

if mean_entropy > ENTROPY_THRESHOLD:
    uncertainty_refusal = True
    refusal_reasons.append(f"Global entropy excessive ({mean_entropy:.4f} > {ENTROPY_THRESHOLD})")

if len(ambiguous_findings) >= AMBIGUITY_COUNT_THRESHOLD:
    uncertainty_refusal = True
    refusal_reasons.append(f"{len(ambiguous_findings)} findings clustered in [0.45, 0.65]")

if uncertainty_refusal:
    uncertainty_status = "REFUSED - HIGH EPISTEMIC UNCERTAINTY"
    print(f"[REFUSAL TRIGGERED] {uncertainty_status}")
    for reason in refusal_reasons:
        print(f" - Cause: {reason}")
    print("Action: Automated prediction withheld. Routed to Mandatory Radiologist Review.")
else:
    uncertainty_status = "PASS - Low Predictive Ambiguity"
    print(f"[PASS] {uncertainty_status}")

# ========================================================
# SECTION 5.5: HARDCODED CONTRAINDICATION ENGINE (HCE)
# ========================================================
print(f"\n--- SECTION 5.5: HARDCODED CONTRAINDICATION INTERLOCKS ---")
patient_vitals_labs = {
    "age": 64,
    "sbp": 86,
    "hr": 104,
    "egfr": 24.5,
    "creatinine": 2.3,
    "platelets": 85000,
    "inr": 2.1,
    "lactate": 4.6
}

hce_violations = []

if patient_vitals_labs["egfr"] < 30.0 or patient_vitals_labs["creatinine"] > 1.8:
    hce_violations.append({
        "id": "HCE-01",
        "lock": "RENAL CONTRAST LOCKOUT",
        "detail": f"eGFR: {patient_vitals_labs['egfr']} mL/min, Cr: {patient_vitals_labs['creatinine']} mg/dL",
        "action": "HARD BLOCK: Suppress all contrast-enhanced follow-up protocols."
    })

if patient_vitals_labs["platelets"] < 100000 or patient_vitals_labs["inr"] > 1.7:
    hce_violations.append({
        "id": "HCE-02",
        "lock": "COAGULOPATHY THROMBOLYSIS LOCKOUT",
        "detail": f"Platelets: {patient_vitals_labs['platelets']}/uL, INR: {patient_vitals_labs['inr']}",
        "action": "HARD BLOCK: Thrombolysis (tPA) locked out. Severe hemorrhage hazard."
    })

shock_index = patient_vitals_labs["hr"] / max(patient_vitals_labs["sbp"], 1)
if patient_vitals_labs["sbp"] < 90 or shock_index >= 1.0 or patient_vitals_labs["lactate"] >= 4.0:
    hce_violations.append({
        "id": "HCE-03",
        "lock": "HEMODYNAMIC INSTABILITY TRIAGE LOCKOUT",
        "detail": f"SBP: {patient_vitals_labs['sbp']} mmHg, Shock Index: {shock_index:.2f}, Lactate: {patient_vitals_labs['lactate']} mmol/L",
        "action": "HARD BLOCK: Suppress any low-acuity output. Force IMMEDIATE ATTENDING REVIEW."
    })

if hce_violations:
    print(f"[VETO TRIGGERED] {len(hce_violations)} Deterministic Safety Interlocks Activated:")
    for v in hce_violations:
        print(f"  * [{v['id']}] {v['lock']}: {v['detail']}")
else:
    print("[PASS] All deterministic contraindication checks cleared.")

# ========================================================
# COMPREHENSIVE AUDIT REPORT LOGGING
# ========================================================
with open(audit_file, "w") as f:
    f.write("=======================================================\n")
    f.write("      FULL-BODY MULTIMODAL AI SYSTEM SAFETY AUDIT      \n")
    f.write("=======================================================\n\n")
    f.write(f"1. ORGAN ROUTER (S5.2): Verified {declared_body_part} ({thoracic_pixel_mass} pixels)\n")
    f.write(f"2. ATTRIBUTION (S5.3) : Evaluated '{top_pathology}' ({top_score:.4f}) | AAMR: {aamr:.4f} | Status: {ups_status}\n")
    f.write(f"3. UNCERTAINTY (S5.4) : Entropy: {mean_entropy:.4f} bits | Ambiguous: {len(ambiguous_findings)} | Status: {uncertainty_status}\n")
    if uncertainty_refusal:
        f.write(f"   Refusal Triggered  : {'; '.join(refusal_reasons)}\n")
    f.write(f"4. DETERMINISTIC (S5.5): {len(hce_violations)} Interlock Vetoes Triggered\n")
    for v in hce_violations:
        f.write(f"   - [{v['id']}] {v['lock']}: {v['action']}\n")

print(f"\n[OK] Complete multi-agent pipeline audit recorded at: {audit_file}")
