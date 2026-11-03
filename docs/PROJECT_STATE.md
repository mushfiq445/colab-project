# SaMD Project State

## 1. Current Status
- **Phase:** Transitioning from Phase 1 Hardening to Task Queue Item 2 (Issue R3).
- **Environment:** Android Termux (git client) -> GitHub (`mushfiq445/colab-project`) -> Google Colab (GPU/CPU execution).
- **Tests Passing:** 99/99 tests passing in Colab (Tier 0, Layers 1-4, Disposition, Audit Log).
- **Audit Verification:** Chained JSONL log verified; head hash confirmed on genuine image runs.

## 2. Completed Milestones
- **Safety Patch 1:** Fail-closed Layer 4 interlocks, YAML single-source config, disposition engine, hash-chained audit log.
- **Task 1 Verified:** Restored M01-M04 Tier 0 pre-checks, aligned HCE-03 Lactate >= 4.0 mmol/L, resolved merge conflicts, and verified real Colab execution.

## 3. Open Issues & Priority Queue
1. [IN PROGRESS] **R3 (Critical): Checkpoint Pipeline & Model Integrity**
   - Quarantine/delete synthetic trainers (`train.py`, `train_cohort.py`, `streaming_dataset.py`).
   - Eliminate `strict=False` and enforce `weights_only=True` in `src/inference.py`.
   - Build a formal Checkpoint Registry (`config/model_registry.yaml` or JSON) recording model ID, SHA-256 weight hash, training dataset provenance, and validation metrics.
   - Record model SHA-256 hash in every audit record.
2. [PENDING] **R4:** Layer 1 weak pixel threshold & Tier 0 integration.
3. [PENDING] **R5:** Layer 3 entropy normalisation / epistemic uncertainty.
4. [PENDING] **R6:** Layer 2 UPS attribution correctness & class-specific masks.
5. [PENDING] **R7/R9:** Housekeeping, unpinned dependencies, data calibration.
6. [PENDING] **36-Agent Expansion:** Spec alignment (S1-S11), backend schemas, async router.

## 4. Pending Decisions & Known Limitations
- `rclone` not installed in runtime environment; local audit logs remain authoritative.
- Seed data runs without clinical profile sidecar fail closed to `HARD_LOCKOUT` by design.
- M02 thoracic pixel minimum temporarily kept at 1,000 px pending Task 3 (R4 calibration).
