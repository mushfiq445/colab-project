import os
import shutil
import subprocess
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchxrayvision as xrv

def load_diagnostic_backbone(device):
    print(f"[BACKBONE] Initializing DenseNet-121 on {device}...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    checkpoint_remote = "mega_pool:best_backbone_checkpoint.pt"
    local_scratch = "/tmp/best_backbone_checkpoint.pt"
    
    # Check if rclone executable is installed and configured
    rclone_path = shutil.which("rclone")
    if rclone_path is not None:
        print(f"[STORAGE] Checking for fine-tuned weights at {checkpoint_remote}...")
        try:
            res = subprocess.run([rclone_path, "copy", checkpoint_remote, "/tmp/"], capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(local_scratch):
                print(f"[STORAGE] Injected fine-tuned checkpoint from {checkpoint_remote}")
                state_dict = torch.load(local_scratch, map_location=device)
                model.load_state_dict(state_dict, strict=False)
            else:
                print(f"[STORAGE] Remote checkpoint not found or rclone copy returned code {res.returncode}. Using baseline weights.")
        except Exception as e:
            print(f"[STORAGE WARNING] Failed to copy remote checkpoint ({e}). Using baseline weights.")
    else:
        print("[STORAGE] rclone executable not detected in PATH. Defaulting to baseline pretrained weights.")
        
    model.to(device)
    model.eval()
    return model

def preprocess_xray(img_path: str, device):
    raw_img = Image.open(img_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor_img = transform(raw_img)
    tensor_img = (tensor_img * 2048.0) - 1024.0
    tensor_img = tensor_img.unsqueeze(0).to(device)
    return tensor_img, raw_img

def run_inference(model, input_tensor, pathologies):
    with torch.no_grad():
        preds = model(input_tensor)
        probs = torch.sigmoid(preds).squeeze().cpu().numpy()
    return {pathologies[i]: float(probs[i]) for i in range(len(pathologies))}
