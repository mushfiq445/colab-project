import numpy as np

def validate_two_factor_organ(dicom_header_body_part: str, lung_mask: np.ndarray, min_pixel_threshold: int = 1000) -> bool:
    header_valid = (dicom_header_body_part or "").strip().upper() == "CHEST"
    thoracic_pixels = int(np.sum(lung_mask > 0))
    anatomy_valid = thoracic_pixels >= min_pixel_threshold

    if not header_valid or not anatomy_valid:
        print(f"[GATE REJECTION] Layer 1 Failed -> DICOM Chest Tag: {header_valid} | Segmented Thoracic Pixels: {thoracic_pixels} (Min: {min_pixel_threshold})")
        return False

    print(f"[GATE PASS] Layer 1 Verified -> Thoracic Pixels: {thoracic_pixels}")
    return True
