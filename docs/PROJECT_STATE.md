# SaMD Project State

## 1. Current Status
- **Phase:** Task 2 (Issue R3: Checkpoint Pipeline & Model Registry) implemented; ready for Colab verification.
- **Environment:** Android Termux -> GitHub (`mushfiq445/colab-project`) -> Google Colab.

## 2. Completed Milestones
- **Safety Patch 1:** Fail-closed Layer 4 interlocks, YAML config, disposition engine, hash-chained audit log.
- **Task 1 Verified:** M01-M04 Tier 0 pre-checks restored, HCE-03 Lactate >= 4.0 mmol/L aligned, 99/99 tests passing in Colab.
- **Task 2 (Issue R3) Implemented:**
  - Quarantined `train.py`, `train_cohort.py`, and `streaming_dataset.py` into `quarantine/`.
  - Added `config/model_registry.yaml` with schema validation.
  - Rewrote `src/inference.py` with `strict=True`, `weights_only=True`, `ModelUnavailable` fail-closed errors, and deterministic weight SHA-256 hashing.
  - Added `model_sha256` and `model_registry_sha256` to `audit_log.jsonl` and `audit_manifest.csv`.
  - Added 14 unit and codebase-hygiene tests in `tests/test_model_registry.py`.

## 3. Open Issues & Priority Queue
1. [NEXT] **R4:** Layer 1 weak pixel threshold (1,000 px vs 100,000 px) & full Tier 0 pre-check wiring in `main.py`.
2. [PENDING] **R5:** Layer 3 uncertainty normalisation & epistemic replacement.
3. [PENDING] **R6:** Layer 2 UPS attribution correctness, class-specific masks & norm5 Grad-CAM upsampling.
4. [PENDING] **R9:** Housekeeping, pinned requirements, rclone fallback, calibration script.
5. [PENDING] **36-Agent Expansion:** Spec alignment (S1-S11), backend schemas, async router.
