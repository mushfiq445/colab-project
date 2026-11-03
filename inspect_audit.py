import os
import pandas as pd

# 1. Inspect the generated audit manifest CSV
manifest_path = "audit_logs/audit_manifest.csv"
if os.path.exists(manifest_path):
    df = pd.read_csv(manifest_path)
    print("\n========== AUDIT MANIFEST SUMMARY ==========")
    print(df.to_string())
else:
    print("[INFO] Audit manifest path not found.")

# 2. Check the Grad-CAM heatmap status
gradcam_file = "/tmp/samd_scratch/gradcam_pneumonia_infiltration.png"
if os.path.exists(gradcam_file):
    print(f"\n[VISUAL AUDIT] Grad-CAM heatmap successfully verified at {gradcam_file}")
else:
    print("[INFO] Grad-CAM output pending.")
