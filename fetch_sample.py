import os
import skimage.io

target_dir = '/content/drive/MyDrive/XRay_Predictions/images'
os.makedirs(target_dir, exist_ok=True)

for item in os.listdir(target_dir):
    p = os.path.join(target_dir, item)
    if os.path.isfile(p):
        os.remove(p)

print("Fetching verified benchmark scan...")
img_url = "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/pneumonia.jpg"
img = skimage.io.imread(img_url)

dest_path = os.path.join(target_dir, "benchmark_xray.png")
skimage.io.imsave(dest_path, img)
print(f"[OK] Successfully saved benchmark scan to: {dest_path}")
