import os
import subprocess
import torch
import skimage.io
import torchxrayvision as xrv

def load_diagnostic_backbone(device, checkpoint_remote="mega_pool:best_backbone_checkpoint.pt"):
    print(f"[BACKBONE] Initializing DenseNet-121 on {device}...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")

    local_scratch = "/tmp/samd_scratch"
    os.makedirs(local_scratch, exist_ok=True)
    local_ckpt = os.path.join(local_scratch, "best_backbone_checkpoint.pt")

    print(f"[STORAGE] Checking for fine-tuned weights at {checkpoint_remote}...")
    res = subprocess.run(["rclone", "copy", checkpoint_remote, local_scratch], capture_output=True, text=True)
    
    if res.returncode == 0 and os.path.exists(local_ckpt):
        try:
            checkpoint = torch.load(local_ckpt, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            loss_val = checkpoint.get("final_loss", "N/A")
            print(f"[WEIGHTS INJECTED] Successfully loaded fine-tuned checkpoint (loss: {loss_val})")
        except Exception as e:
            print(f"[WARNING] Failed to inject custom weights: {e}. Falling back to baseline.")
    else:
        print("[BASELINE WEIGHTS] Checkpoint not found in pool. Using stock TorchXRayVision weights.")

    model = model.to(device)
    model.eval()
    return model

def preprocess_xray(img_path: str, device):
    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255)
    
    if len(img.shape) > 2:
        img = img.mean(2)
        
    img = img[None, ...]
    raw_img = img[0]
    
    img_tensor = torch.from_numpy(img).float()
    transform = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    img_tensor = transform.squeeze(0).to(device)
    
    return img_tensor, raw_img

def run_inference(model, input_tensor, pathologies):
    with torch.no_grad():
        preds = model(input_tensor)
        preds = torch.sigmoid(preds)
        
    prob_dict = {path: float(preds[0][i].cpu().item()) for i, path in enumerate(pathologies)}
    return prob_dict
