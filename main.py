import os
import numpy as np
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam

# Paths
drive_root = '/content/drive/MyDrive/XRay_Predictions'
images_dir = os.path.join(drive_root, 'images')
output_file = os.path.join(drive_root, 'predictions.txt')
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

# 5. Generate Grad-CAM for the highest scoring pathology
top_pathology, top_score = sorted_results[0]
print(f"\nGenerating Grad-CAM heatmap for top condition: {top_pathology} ({top_score:.4f})...")

target_idx = model.pathologies.index(top_pathology)
target_layer = model.model.features.denseblock4
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

gradcam_save_path = os.path.join(drive_root, f"gradcam_{top_pathology.lower().replace(' ', '_')}.png")
plt.tight_layout()
plt.savefig(gradcam_save_path, bbox_inches='tight', dpi=150)
plt.close()

print(f"[OK] Saliency map saved to: {gradcam_save_path}")
