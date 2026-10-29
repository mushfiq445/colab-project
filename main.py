import os
import numpy as np
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv

# Set up paths in Google Drive
drive_root = '/content/drive/MyDrive/XRay_Predictions'
images_dir = os.path.join(drive_root, 'images')
output_file = os.path.join(drive_root, 'predictions.txt')
os.makedirs(images_dir, exist_ok=True)

# 1. Load the pre-trained DenseNet model
print("Loading DenseNet model...")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# 2. Check for any uploaded files inside the images folder
candidates = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if not f.startswith('.')]

img_loaded = False
if candidates:
    target_img_path = candidates[0]
    print(f"Detected file: {target_img_path}")
    try:
        raw_img = skimage.io.imread(target_img_path)
        
        # Handle multichannel/RGBA images by reducing to 2D grayscale
        if raw_img.ndim == 3:
            raw_img = raw_img.mean(axis=2)
            
        max_val = 65535 if raw_img.dtype == np.uint16 else 255
        img = xrv.datasets.normalize(raw_img, max_val)
        img = img[None, ...]
        img_loaded = True
        print(f"Successfully loaded and preprocessed image: {os.path.basename(target_img_path)}")
    except Exception as e:
        print(f"Failed to read image ({e}), falling back to simulation array.")

if not img_loaded:
    print("Running simulation array...")
    np.random.seed(42)
    raw_img = np.random.randint(50, 200, size=(512, 512), dtype=np.uint8)
    img = xrv.datasets.normalize(raw_img, 255)[None, ...]

# 3. Transform to 224x224 input required by DenseNet
transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])
img_tensor = transform(img)
img_tensor = torch.from_numpy(img_tensor)

# 4. Run inference
print("Running inference on image...")
with torch.no_grad():
    predictions = model(img_tensor[None, ...])

# 5. Output and save results
results = dict(zip(model.pathologies, predictions[0].numpy()))

print("\n--- PREDICTION RESULTS ---")
with open(output_file, 'w') as f:
    f.write("--- PREDICTION RESULTS ---\n")
    for pathology, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        line = f"{pathology:25s}: {score:.4f}"
        print(line)
        f.write(line + "\n")

print(f"\n[OK] Results successfully saved to: {output_file}")
