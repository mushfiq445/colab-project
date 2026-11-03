# SaMD Project State

## 1. Current Status
- **Phase:** Task 3 (Issue R4: Layer 1 & Tier 0 Wiring) implemented; ready for Colab verification.
- **Environment:** Android Termux -> GitHub (`mushfiq445/colab-project`) -> Google Colab (GPU/CPU).

## 2. Completed Milestones
- **Safety Patch 1:** Fail-closed Layer 4 interlocks, YAML config, disposition engine, hash-chained audit log.
- **Task 1 Verified:** M01-M04 Tier 0 pre-checks restored, HCE-03 Lactate >= 4.0 mmol/L aligned.
- **Task 2 Verified (R3):** Checkpoint registry (`model_registry.yaml`), `weights_only=True`, `strict=True`, hash tracking.
- **Task 3 Implemented (R4):** 
  - Created `config/organ_thresholds.yaml` enforcing 100,000 px minimum total, 30,000 px per lung.
  - Replaced manual wrapping in `main.py` with the full `execute_tier0_precheck` orchestration.
  - Standard images correctly labeled `VISUAL_ONLY` instead of fabricating DICOM headers.

## 3. Open Issues & Priority Queue
1. [NEXT] **R5:** Layer 3 entropy normalisation normalise by class count & replace/augment with epistemic uncertainty.
2. [PENDING] **R6:** Layer 2 UPS attribution correctness, class-specific masks & norm5 Grad-CAM upsampling.
3. [PENDING] **R9:** Housekeeping, unpinned dependencies, rclone fallback, calibration script.
4. [PENDING] **36-Agent Expansion:** Spec alignment (S1-S11), backend schemas, async router.
