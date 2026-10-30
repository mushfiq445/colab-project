import os
import subprocess

class HybridStorageRouter:
    def __init__(self):
        self.gdrive_base = "gdrive:XRay_Predictions"
        self.mega_pool_base = "mega_pool:"

    def push(self, remote_target: str, local_path: str) -> bool:
        cmd = ["rclone", "copy", local_path, remote_target]
        res = subprocess.run(cmd, capture_output=True, text=True)
        name = os.path.basename(local_path)
        if res.returncode == 0:
            print(f"[SYNC SUCCESS] {name} -> {remote_target}")
            return True
        print(f"[SYNC FAILED] {name} -> {remote_target} | Error: {res.stderr.strip()}")
        return False

    def route_file(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # 1. Heavy files, models, archives, and bulk images -> Unified 3-Mega Pool
        if ext in [".pdf", ".zip", ".tar", ".gz", ".pt", ".onnx", ".bin", ".jpg", ".jpeg", ".png", ".dcm"] or size_mb >= 5.0:
            return self.push(self.mega_pool_base, file_path)

        # 2. Critical manifests, sidecar telemetry, text audits (< 5 MB) -> Google Drive
        else:
            return self.push(self.gdrive_base, file_path)
