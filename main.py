import os
import numpy as np
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv

# Set up output path directly to mounted Google Drive
output_dir = '/content/drive/MyDrive/XRay_Predictions'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'predictions.txt')

# 1. Load the pre-trained DenseNet model
print("Loading DenseNet model...")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# 2. Generate a valid synthetic 8-bit chest X-ray simulation array (512x512)
np.random.seed(42)
img = np.random.randint(50, 200, size=(512, 512), dtype=np.uint8)

# 3. Preprocess: normalize 8-bit pixels [0, 255] to [-1024, 1024]
img = xrv.datasets.normalize(img, 255)

# Ensure 2D grayscale shape: (1, height, width)
if len(img.shape) > 2:
    img = img.mean(2)
img = img[None, ...]

# 4. Crop and resize to 224x224 expected by DenseNet
transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])
img_tensor = transform(img)
img_tensor = torch.from_numpy(img_tensor)

# 5. Run inference (Batch shape: [1, 1, 224, 224])
print("Running inference...")
with torch.no_grad():
    predictions = model(img_tensor[None, ...])

# 6. Display results and save them directly to Google Drive
results = dict(zip(model.pathologies, predictions[0].numpy()))

print("\n--- PREDICTION RESULTS ---")
with open(output_file, 'w') as f:
    f.write("--- PREDICTION RESULTS ---\n")
    for pathology, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        line = f"{pathology:25s}: {score:.4f}"
        print(line)
        f.write(line + "\n")

print(f"\n[OK] Results successfully saved to: {output_file}")
