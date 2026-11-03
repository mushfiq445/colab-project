# SaMD Project State

## 1. Current Status
- **Phase:** Moving to Task Queue Item 3 (Issue R4: Layer 1 & Tier 0 Wiring).
- **Environment:** Android Termux -> GitHub (`mushfiq445/colab-project`) -> Google Colab (GPU/CPU).
- **Tests Passing:** 113/113 passing in Colab.
- **Audit Verification:** Hash-chained audit logs cryptographically verified with deterministic model weight SHA-256 (`033badcf...`).

## 2. Completed Milestones
- **Safety Patch 1:** Fail-closed Layer 4 interlocks, YAML single-source config, disposition engine, hash-chained audit log.
- **Task 1 Verified:** M01-M04 Tier 0 pre-checks reconstructed, HCE-03 Lactate >= 4.0 mmol/L aligned, full Colab execution verified.
- **Task 2 (Issue R3) Verified:**
  - Quarantined synthetic trainers into `quarantine/`.
  - Added `config/model_registry.yaml` with schema validation.
  - Rewrote `src/inference.py` with `strict=True`, `weights_only=True`, and deterministic weight SHA-256 hashing.
  - Recorded model provenance in `audit_log.jsonl` and `audit_manifest.csv`.
  - 14/14 tests in `tests/test_model_registry.py` passed in Colab.

## 3. Open Issues & Priority Queue
1. [IN PROGRESS] **R4: Layer 1 Weak Pixel Threshold & Tier 0 Wiring in `main.py`**
   - Replace hardcoded `"CHEST"` in `main.py` with real metadata routing via `execute_tier0_precheck`.
   - Wire M01 (DICOM validator), M03 (abdomen veto), and M04 (ECG gate) into pipeline execution.
   - Move organ thresholds from code into a unified YAML config (`config/organ_thresholds.yaml`).
   - Calibrate thoracic pixel threshold from 1,000 px to 100,000 px minimum (~38% of 512x512 grid) plus per-lung minimums.
   - Label standard 2D image inputs without DICOM headers as `VISUAL_ONLY`.
2. [PENDING] **R5:** Layer 3 entropy normalisation & epistemic uncertainty replacement.
3. [PENDING] **R6:** Layer 2 UPS attribution correctness, class-specific masks & norm5 Grad-CAM upsampling.
4. [PENDING] **R7/R9:** Housekeeping, unpinned dependencies, rclone fallback, calibration script.
5. [PENDING] **36-Agent Expansion:** Spec alignment (S1-S11), backend schemas, async router.
