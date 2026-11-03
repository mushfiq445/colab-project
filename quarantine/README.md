# QUARANTINED MODULES — DO NOT IMPORT IN PRODUCTION

The scripts in this directory (`train.py`, `train_cohort.py`, `streaming_dataset.py`)
were quarantined under Issue R3:
1. `train.py` trained on the model's own outputs without ground truth.
2. `train_cohort.py` trained on synthetic Gaussian noise with random labels and wrote to the active checkpoint path.

Per Non-Negotiable Rule 2 & Rule 8:
- Synthetic data may NEVER be used to train a checkpoint that production can load.
- Nothing in `src/` or `main.py` may import from `quarantine/`.
