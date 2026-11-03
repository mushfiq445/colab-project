import numpy as np

def calculate_aamr(attribution_map: np.ndarray, lung_mask: np.ndarray) -> float:
    """
    Calculates Anatomical Attribution Mass Ratio (AAMR).
    """
    total_attr = np.sum(attribution_map)
    if total_attr == 0:
        return 0.0

    if attribution_map.shape != lung_mask.shape:
        # Resize mask if shapes differ
        from skimage.transform import resize
        lung_mask = resize(lung_mask, attribution_map.shape, order=0, preserve_range=True).astype(np.uint8)

    in_mask_attr = np.sum(attribution_map * lung_mask)
    return float(in_mask_attr / total_attr)

def compute_gradcam(model, target_layer, input_tensor, target_class: int):
    from captum.attr import LayerGradCam
    grad_cam = LayerGradCam(model, target_layer)
    attr = grad_cam.attribute(input_tensor, target=target_class)
    attr_np = attr.squeeze().detach().cpu().numpy()
    attr_np = np.maximum(attr_np, 0)
    if np.max(attr_np) > 0:
        attr_np = attr_np / np.max(attr_np)
    return attr_np

def evaluate_ups_gate(model, target_layer, input_tensor, top_class_idx: int,
                      lung_mask: np.ndarray, diagnostic_threshold: float = 0.50,
                      aamr_threshold: float = 0.85, top_score: float = 0.0):
    if top_score < diagnostic_threshold:
        return None, 1.0, "Bypassed (Unremarkable Scan: Score below threshold)"

    attr_map = compute_gradcam(model, target_layer, input_tensor, top_class_idx)
    aamr = calculate_aamr(attr_map, lung_mask)

    if aamr < aamr_threshold:
        return attr_map, aamr, f"REJECTED (AAMR {aamr:.3f} < {aamr_threshold})"

    return attr_map, aamr, f"PASSED (AAMR {aamr:.3f} >= {aamr_threshold})"
