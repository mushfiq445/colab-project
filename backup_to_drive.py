import os
import shutil
from google.colab import drive

drive.mount('/content/drive', force_remount=False)

gdrive_dest = '/content/drive/MyDrive/XRay_Predictions'
os.makedirs(f"{gdrive_dest}/audit_logs", exist_ok=True)
os.makedirs(f"{gdrive_dest}/scratch_backup", exist_ok=True)

if os.path.exists("audit_logs"):
    for item in os.listdir("audit_logs"):
        s = os.path.join("audit_logs", item)
        d = os.path.join(f"{gdrive_dest}/audit_logs", item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print("[SUCCESS] Audit logs synced to Google Drive!")

scratch_path = "/tmp/samd_scratch"
if os.path.exists(scratch_path):
    for item in os.listdir(scratch_path):
        s = os.path.join(scratch_path, item)
        d = os.path.join(f"{gdrive_dest}/scratch_backup", item)
        if os.path.isfile(s):
            shutil.copy2(s, d)
    print("[SUCCESS] Model checkpoints and heatmaps synced to Google Drive!")

print("\nAll files are safely stored in Google Drive under: XRay_Predictions")
