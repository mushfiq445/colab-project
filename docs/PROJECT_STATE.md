# SaMD Project State

## 1. Current Status
- **Phase:** Task 6 (Issue R9: Housekeeping) implemented; preparing for calibration.
- **Environment:** Android Termux -> GitHub -> Google Colab.
- **Test Status:** Passing 113/113.

## 2. Completed Milestones
- **Safety Patch 1:** Fail-closed Layer 4 interlocks, YAML config, disposition engine, hash-chained audit log.
- **Task 1 (R1/R2):** M01-M04 Tier 0 pre-checks restored, HCE-03 Lactate >= 4.0 mmol/L aligned.
- **Task 2 (R3):** Checkpoint registry (`model_registry.yaml`), `weights_only=True`, `strict=True`, hash tracking.
- **Task 3 (R4):** 100,000 px thoracic thresholds in YAML, M01-M03 orchestrated, `VISUAL_ONLY` routing.
- **Task 4 (R5):** Fixed Layer 3 predictive entropy dilution (max class-wise binary cross-entropy).
- **Task 5 (R6):** Implemented pathology-specific Layer 2 UPS gating and bilinear Grad-CAM upsampling (norm5).

## 3. Open Issues & Priority Queue
- **Task 6 (R9):** Pinned requirements.txt, added graceful rclone fallback.
1. [NEXT] **Task 6.3:** Threshold Calibration Script (`scripts/calibrate_thresholds.py`).
2. [PENDING] **Task 7:** 36-Agent Expansion: Spec alignment (S1-S11), backend schemas, async router.
