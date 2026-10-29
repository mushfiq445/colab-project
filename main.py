import os
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

# 1. Load DenseNet model
print("Loading DenseNet model...")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# 2. Find and load image
candidates = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if not f.startswith('.')]
img_loaded = False

if candidates:
    target_img_path = candidates[0]
    print(f"Detected file: {target_img_path}")
    try:
        raw_img = skimage.io.imread(target_img_path)
        if raw_img.ndim == 3:
            raw_img = raw_img.mean(axis=2)
        max_val = 65535 if raw_img.dtype == np.uint16 else 255
        img = xrv.datasets.normalize(raw_img, max_val)[None, ...]
        img_loaded = True
        print(f"Successfully loaded: {os.path.basename(target_img_path)}")
    except Exception as e:
        print(f"Failed to load image ({e}), falling back to simulation array.")

if not img_loaded:
    print("Running simulation array...")
    np.random.seed(42)
    raw_img = np.random.randint(50, 200, size=(512, 512), dtype=np.uint8)
    img = xrv.datasets.normalize(raw_img, 255)[None, ...]

# 3. Transform to 224x224
transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])
img_tensor = torch.from_numpy(transform(img))

# 4. Run inference
print("Running inference...")
with torch.no_grad():
    predictions = model(img_tensor[None, ...])

results = dict(zip(model.pathologies, predictions[0].numpy()))
sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

print("\n--- PREDICTION RESULTS ---")
with open(output_file, 'w') as f:
    f.write("--- PREDICTION RESULTS ---\n")
    for pathology, score in sorted_results:
        line = f"{pathology:25s}: {score:.4f}"
        print(line)
        f.write(line + "\n")

print(f"\n[OK] Predictions saved to: {output_file}")

# 5. Generate Grad-CAM for Effusion
top_pathology = "Effusion"
print(f"\nGenerating Grad-CAM heatmap for {top_pathology}...")

target_idx = model.pathologies.index(top_pathology)
target_layer = model.features.denseblock4
gradcam = LayerGradCam(model, target_layer)

input_tensor = img_tensor.clone().detach().unsqueeze(0).requires_grad_(True)
attribution = gradcam.attribute(input_tensor, target=target_idx)

upsampled_attr = LayerGradCam.interpolate(attribution, (224, 224), interpolate_mode="bilinear")
heatmap = upsampled_attr.squeeze().detach().cpu().numpy()
heatmap = np.maximum(heatmap, 0)
if heatmap.max() > 0:
    heatmap = heatmap / heatmap.max()

raw_display_img = input_tensor.squeeze().detach().cpu().numpy()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(raw_display_img, cmap="gray")
axs[0].set_title("Input Chest X-Ray", fontsize=12)
axs[0].axis("off")

axs[1].imshow(raw_display_img, cmap="gray")
axs[1].imshow(heatmap, cmap="jet", alpha=0.45)
axs[1].set_title(f"Grad-CAM: {top_pathology}", fontsize=12)
axs[1].axis("off")

gradcam_save_path = os.path.join(drive_root, "gradcam_effusion.png")
plt.tight_layout()
plt.savefig(gradcam_save_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"[OK] Saliency map saved to: {gradcam_save_path}")

# ==========================================
# SECTION 5.3: AUTOMATED SAFETY GATE (UPS)
# ==========================================
print("\nRunning Unexplainable Prediction Suppression (UPS) check...")

seg_model = xrv.baseline_models.chestx_det.PSPNet()

with torch.no_grad():
    seg_mask = seg_model(input_tensor.detach())

left_lung_idx = seg_model.targets.index("Left Lung")
right_lung_idx = seg_model.targets.index("Right Lung")

lung_mask = (seg_mask[0, left_lung_idx] > 0.5) | (seg_mask[0, right_lung_idx] > 0.5)
lung_mask_np = lung_mask.cpu().numpy().astype(np.float32)

lung_mask_resized = skimage.transform.resize(
    lung_mask_np, 
    heatmap.shape,
    order=0,
    anti_aliasing=False
)

total_heatmap_mass = np.sum(heatmap)
anatomical_mass = np.sum(heatmap * lung_mask_resized)

if total_heatmap_mass > 0:
    aamr = anatomical_mass / total_heatmap_mass
else:
    aamr = 0.0

print(f"Anatomical Attribution Mass Ratio (AAMR): {aamr:.4f}")

AAMR_THRESHOLD = 0.85

if aamr >= AAMR_THRESHOLD:
    status = "VERIFIED - Pathologically Plausible"
    print(f"[PASS] {status}: Attribution aligns with pulmonary anatomy.")
else:
    status = "REJECTED - Untrustworthy Shortcut Detected"
    print(f"[ALERT] {status}: Heatmap mass falls outside lung boundaries.")
    print("Action: Automated diagnostic label SUPPRESSED. Case routed to human review.")

with open(audit_file, "w") as f:
    f.write("--- SECTION 5.3: SAFETY GATE (UPS) AUDIT REPORT ---\n")
    f.write(f"Condition Evaluated: {top_pathology}\n")
    f.write(f"Raw Score: {results[top_pathology]:.4f}\n")
    f.write(f"Calculated AAMR: {aamr:.4f}\n")
    f.write(f"Safety Status: {status}\n")

# ========================================================
# SECTION 5.4: EPISTEMIC UNCERTAINTY & ENTROPY REFUSAL
# ========================================================
print("\nRunning Epistemic Uncertainty & Entropy Audit...")

raw_probs = np.array(list(results.values()), dtype=np.float64)
probs_clipped = np.clip(raw_probs, 1e-7, 1.0 - 1e-7)
class_entropies = -(probs_clipped * np.log2(probs_clipped) + (1.0 - probs_clipped) * np.log2(1.0 - probs_clipped))

mean_entropy = float(np.mean(class_entropies))
print(f"Mean Predictive Entropy: {mean_entropy:.4f} bits (Max possible: 1.0000)")

ambiguous_findings = [path for path, score in results.items() if 0.45 <= score <= 0.65]
print(f"Number of ambiguous findings in [0.45, 0.65]: {len(ambiguous_findings)}")

ENTROPY_THRESHOLD = 0.85
AMBIGUITY_COUNT_THRESHOLD = 5

uncertainty_refusal = False
refusal_reasons = []

if mean_entropy > ENTROPY_THRESHOLD:
    uncertainty_refusal = True
    refusal_reasons.append(f"Excessive global entropy ({mean_entropy:.4f} > {ENTROPY_THRESHOLD})")

if len(ambiguous_findings) >= AMBIGUITY_COUNT_THRESHOLD:
    uncertainty_refusal = True
    refusal_reasons.append(f"{len(ambiguous_findings)} classes clustered in ambiguous range [0.45, 0.65]")

if uncertainty_refusal:
    uncertainty_status = "REFUSED - HIGH EPISTEMIC UNCERTAINTY"
    print(f"\n[REFUSAL TRIGGERED] {uncertainty_status}")
    for reason in refusal_reasons:
        print(f" - Cause: {reason}")
    print("Action: Output suppressed. Routed to MANDATORY URGENT RADIOLOGIST REVIEW.")
else:
    uncertainty_status = "PASS - Low Predictive Ambiguity"
    print(f"[PASS] {uncertainty_status}")

with open(audit_file, "a") as f:
    f.write(f"\n--- SECTION 5.4: UNCERTAINTY & ENTROPY AUDIT ---\n")
    f.write(f"Mean Predictive Entropy: {mean_entropy:.4f}\n")
    f.write(f"Ambiguous Findings Count: {len(ambiguous_findings)}\n")
    f.write(f"Uncertainty Status: {uncertainty_status}\n")
    if uncertainty_refusal:
        f.write(f"Refusal Reasons: {'; '.join(refusal_reasons)}\n")

print(f"\n[OK] Combined safety audit successfully updated at: {audit_file}")
