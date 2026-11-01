import os
import subprocess

class HybridStorageRouter:
    """
    Handles hybrid storage routing between local scratch, Mega pool, and Google Drive.
    Falls back gracefully if rclone is unconfigured or unavailable.
    """
    def __init__(self):
        self.mega_pool_base = "mega_pool:"
        self.gdrive_base = "gdrive:XRay_Predictions"

    def push(self, remote_target, file_path):
        if not os.path.exists(file_path):
            print(f"[STORAGE WARNING] File not found for routing: {file_path}")
            return False
            
        # Check if rclone is available and configured
        config_check = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
        if config_check.returncode != 0:
            print(f"[STORAGE] Rclone remotes not configured. Skipping remote sync for {file_path}.")
            return False

        cmd = ["rclone", "copyto", file_path, f"{remote_target}/{os.path.basename(file_path)}"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"[SYNC SUCCESS] {file_path} -> {remote_target}")
                return True
            else:
                print(f"[SYNC NOTICE] Local fallback used for {file_path} (rclone code {res.returncode})")
                return False
        except Exception as e:
            print(f"[STORAGE ERROR] Failed to route {file_path}: {e}")
            return False

    def route_file(self, file_path):
        if "checkpoint" in file_path or "gradcam" in file_path:
            return self.push(self.mega_pool_base, file_path)
        else:
            return self.push(self.gdrive_base, file_path)
