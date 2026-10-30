import torch
import torchvision.transforms as transforms
import skimage.io
import torchxrayvision as xrv
import numpy as np

def load_diagnostic_backbone(device: torch.device):
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device)
    model.eval()
    return model

def preprocess_xray(image_path: str, device: torch.device):
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)

    if len(img.shape) > 2:
        img = img.mean(axis=2)

    img = img[np.newaxis, :, :]

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)

    tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
    tensor.requires_grad = True
    return tensor, img.squeeze()

def run_inference(model, tensor: torch.Tensor, pathologies: list):
    with torch.no_grad():
        preds = model(tensor).cpu().numpy()[0]
    return {pathologies[i]: float(preds[i]) for i in range(len(pathologies))}
