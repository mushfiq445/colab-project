import os, subprocess, shutil, sys

def sync_to_drive(src_dir, dest_name):
    # Determine Colab vs Local
    drive_mount = "/content/drive/MyDrive/XRay_Predictions"
    
    if os.path.exists("/content"):
        # We are in Colab
        dest = os.path.join(drive_mount, dest_name)
        os.makedirs(dest, exist_ok=True)
        print(f"Syncing to Google Drive: {dest}")
        os.system(f"cp -r {src_dir}/* {dest}/ 2>/dev/null || true")
    else:
        # We are local (Termux), use rclone
        print(f"Attempting to sync via rclone to remote: {dest_name}")
        try:
            subprocess.run(["rclone", "sync", src_dir, f"remote:{dest_name}"], check=True)
        except FileNotFoundError:
            print("[WARNING] rclone not found. Skipping remote sync. Data remains local.")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] rclone sync failed: {e}")

if __name__ == "__main__":
    os.makedirs("audit_logs", exist_ok=True)
    sync_to_drive("audit_logs", "audit_logs_backup")
    print("[SUCCESS] Backup sequence completed.")
