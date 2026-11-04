# SaMD Project State

## 1. Current Status
- **Phase:** Task 4 (Issue R5: Layer 3 Uncertainty Normalisation) implemented; ready for Colab verification.
- **Environment:** Android Termux -> GitHub (`mushfiq445/colab-project`) -> Google Colab (GPU/CPU).

## 2. Completed Milestones
- **Safety Patch 1:** Fail-closed Layer 4 interlocks, YAML config, disposition engine, hash-chained audit log.
- **Task 1 Verified:** M01-M04 Tier 0 pre-checks restored, HCE-03 Lactate >= 4.0 mmol/L aligned.
- **Task 2 Verified (R3):** Checkpoint registry (`model_registry.yaml`), `weights_only=True`, `strict=True`, hash tracking.
- **Task 3 Verified (R4):** 100,000 px thoracic thresholds in YAML, M01-M03 orchestrated, `VISUAL_ONLY` routing.
- **Task 4 Implemented (R5):** Fixed Layer 3 predictive entropy dilution by enforcing max class-wise binary cross-entropy.

## 3. Open Issues & Priority Queue
1. [NEXT] **Task 5 (R6):** Layer 2 UPS attribution correctness, class-specific masks & norm5 Grad-CAM upsampling.
2. [PENDING] **Task 6 (R9):** Housekeeping, unpinned dependencies, rclone fallback, calibration script.
3. [PENDING] **Task 7:** 36-Agent Expansion: Spec alignment (S1-S11), backend schemas, async router.
