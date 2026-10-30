import os
import sys
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import skimage.io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage_router import HybridStorageRouter
import torchxrayvision as xrv

class ClinicalXRayDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        # Load grayscale image as numpy array
        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)  # Scale to [-1024, 1024]
        
        # Check color channels
        if len(img.shape) > 2:
            img = img.mean(2)
            
        # Add channel dimension (1, H, W)
        img = img[None, ...]
        
        # Resize to 224x224 using bilinear interpolation
        img_tensor = torch.from_numpy(img).float()
        transform = nn.functional.interpolate(img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        img_tensor = transform.squeeze(0)
            
        return img_tensor, path

def build_training_pipeline(device, learning_rate=1e-4):
    print(f"[MODEL INIT] Loading DenseNet-121 diagnostic backbone on {device}...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # Freeze lower feature blocks
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Unfreeze dense block 4 and classifier head
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-2
    )
    criterion = nn.BCEWithLogitsLoss()
    
    return model, optimizer, criterion

def run_training_loop(epochs=5, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE SETUP] Training compute backend: {device}")
    
    router = HybridStorageRouter()
    
    data_dir = "/content/drive/MyDrive/XRay_Predictions/images"
    if not os.path.exists(data_dir):
        data_dir = "audit_logs/images"
        
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + \
                  glob.glob(os.path.join(data_dir, "*.png"))
                  
    if not image_paths:
        print(f"[ERROR] No training images located in {data_dir}.")
        return

    print(f"[DATASET] Loaded {len(image_paths)} scan(s) for training.")

    dataset = ClinicalXRayDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    model, optimizer, criterion = build_training_pipeline(device)
    
    scratch_dir = "/tmp/samd_scratch"
    os.makedirs(scratch_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(scratch_dir, "best_backbone_checkpoint.pt")

    print("\n========== INITIATING FINE-TUNING CYCLE ==========")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step, (batch_imgs, _) in enumerate(dataloader):
            batch_imgs = batch_imgs.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_imgs)
            
            target = torch.sigmoid(logits.detach())
            loss = criterion(logits, target)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / max(len(dataloader), 1)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    print("\n[TRAINING COMPLETE] Serializing checkpoint...")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_loss": avg_loss
    }, best_checkpoint_path)

    print(f"[LOCAL CHECKPOINT] Staged at {best_checkpoint_path}")
    print("[CLOUD SYNC] Dispatching model weights to Mega union pool...")
    router.route_file(best_checkpoint_path)

if __name__ == "__main__":
    run_training_loop()
