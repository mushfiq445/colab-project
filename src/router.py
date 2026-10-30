import torch
import numpy as np
import skimage.io
import torchvision.transforms as transforms
import torchxrayvision as xrv

def load_pspnet_segmenter(device: torch.device):
    model = xrv.baseline_models.chestx_det.PSPNet()
    model = model.to(device)
    model.eval()
    return model

def segment_thoracic_anatomy(image_path: str, segmenter_model, device: torch.device):
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)

    if len(img.shape) > 2:
        img = img.mean(axis=2)

    img = img[np.newaxis, :, :]

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(512)
    ])
    img_512 = transform(img)

    tensor_512 = torch.from_numpy(img_512).unsqueeze(0).float().to(device)

    with torch.no_grad():
        seg_output = segmenter_model(tensor_512)
        pred_mask = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()

    thoracic_mask = (pred_mask > 0).astype(np.uint8)
    thoracic_pixels = int(np.sum(thoracic_mask))

    return thoracic_mask, thoracic_pixels

def validate_two_factor_organ(dicom_header_body_part: str, thoracic_pixels: int, min_pixel_threshold: int = 1000) -> bool:
    header_valid = (dicom_header_body_part or "").strip().upper() == "CHEST"
    anatomy_valid = thoracic_pixels >= min_pixel_threshold

    print(f"Declared Metadata Anatomy : {dicom_header_body_part}")
    print(f"Visual Pixel Anatomy      : {"CHEST" if anatomy_valid else "NON-CHEST"} (Thoracic pixels: {thoracic_pixels})")

    if not header_valid or not anatomy_valid:
        print(f"[GATE REJECTION] Layer 1 Failed -> DICOM Valid: {header_valid} | Anatomy Valid: {anatomy_valid}")
        return False

    print("[PASS] Anatomy Verified: Proceeding to diagnostic inference.")
    return True
