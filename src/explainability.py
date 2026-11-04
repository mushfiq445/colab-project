"""Layer 2: Unexplainable Prediction Suppression (UPS) Gate."""
import numpy as np
import torch
from typing import Any, Tuple, Optional
from torch.nn.functional import interpolate

def compute_gradcam(
    model: Any,
    target_layer: Any,
    input_tensor: torch.Tensor,
    target_class_idx: int,
) -> np.ndarray:
    try:
        from captum.attr import LayerGradCam
    except ImportError:
        raise RuntimeError("Captum is required for UPS gating.")

    model.eval()
    gradcam = LayerGradCam(model, target_layer)
    attr = gradcam.attribute(input_tensor, target=target_class_idx)
    
    # Task 5/Issue R6: Use bilinear interpolation (norm5) to upsample 7x7 feature map back to 224x224
    upsampled = interpolate(attr, size=(224, 224), mode="bilinear", align_corners=False)
    
    # Detach, squeeze, and normalize 0-1
    heatmap = upsampled.squeeze().detach().cpu().numpy()
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap

def evaluate_ups_gate(
    model: Any,
    target_layer: Any,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    pathology_name: str,
    organ_masks: dict,
    diagnostic_threshold: float = 0.30,
    aamr_threshold: float = 0.85,
    top_score: float = 0.0,
) -> Tuple[Optional[np.ndarray], float, str]:
    if top_score < diagnostic_threshold:
        return None, 0.0, "PASSED_BELOW_DIAGNOSTIC_THRESHOLD"

    heatmap = compute_gradcam(model, target_layer, input_tensor, target_class_idx)
    
    # Task 5/Issue R6: Pathology-specific mask attribution
    # Instead of a generic mask, match specific pathologies to specific anatomical bounds
    if pathology_name in ["Pleural_Thickening", "Effusion"]:
        # Should heavily activate at the edges/bottom of the lungs
        mask = organ_masks.get("thorax_union", np.ones_like(heatmap, dtype=bool))
    elif pathology_name in ["Cardiomegaly"]:
        # Should activate inside the Heart mask
        mask = organ_masks.get("Heart", np.ones_like(heatmap, dtype=bool))
    else:
        # Generic lung fields for Pneumonia, Infiltration, etc.
        mask_l = organ_masks.get("Left Lung", np.zeros_like(heatmap, dtype=bool))
        mask_r = organ_masks.get("Right Lung", np.zeros_like(heatmap, dtype=bool))
        mask = mask_l | mask_r

    # Resize mask to 224x224 to match input_tensor and Grad-CAM
    import cv2
    mask_224 = cv2.resize(mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Calculate AAMR: Sum of attribution inside mask / Total attribution
    in_mask_attr = np.sum(heatmap[mask_224])
    total_attr = np.sum(heatmap)
    
    aamr = float(in_mask_attr / (total_attr + 1e-9))
    
    if aamr < aamr_threshold:
        return heatmap, aamr, "L2_UPS_REJECTED"
        
    return heatmap, aamr, "PASSED"


def calculate_aamr(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Legacy helper for test_safety_gates.py compatibility."""
    import cv2
    mask_res = cv2.resize(mask.astype(np.uint8), (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return float(np.sum(heatmap[mask_res]) / (np.sum(heatmap) + 1e-9))
