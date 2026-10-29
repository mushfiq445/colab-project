import torch
import numpy as np
from captum.attr import LayerGradCam

def calculate_aamr(attribution_map: np.ndarray, lung_mask: np.ndarray) -> float:
    pos_attr = np.maximum(0, attribution_map)
    total_mass = np.sum(pos_attr)
    if total_mass == 0:
        return 0.0
    in_mask_mass = np.sum(pos_attr * (lung_mask > 0))
    return float(in_mask_mass / total_mass)

def evaluate_ups_gate(model, target_layer, input_tensor: torch.Tensor, target_class_idx: int, 
                      lung_mask: np.ndarray, diagnostic_threshold: float = 0.50, 
                      aamr_threshold: float = 0.85, top_score: float = 0.0):
    if top_score < diagnostic_threshold:
        return True, 1.0, "Bypassed (Unremarkable Scan: Score below threshold)"

    lgc = LayerGradCam(model, target_layer)
    attr = lgc.attribute(input_tensor, target=target_class_idx)
    attr_np = attr.squeeze().detach().cpu().numpy()

    if attr_np.shape != lung_mask.shape:
        import skimage.transform
        attr_np = skimage.transform.resize(attr_np, lung_mask.shape, mode="reflect", anti_aliasing=True)

    aamr = calculate_aamr(attr_np, lung_mask)
    passed = aamr >= aamr_threshold
    status = f"Passed (AAMR {aamr:.3f} >= {aamr_threshold})" if passed else f"Suppressed (AAMR {aamr:.3f} < {aamr_threshold})"
    return passed, aamr, status
