"""Stage M04 - 12-Lead ECG Biosignal Quality & Preprocessing Gate.

Enforces:
- 12-lead matrix geometry (12, N)
- Sampling frequency validation (default 500 Hz, 10 seconds -> 5,000 samples)
- Flatline / lead-off / NaN / rails saturation detection
- 4th-order Butterworth 0.5-40 Hz bandpass filtering
- Pooled Signal-to-Noise Ratio (SNR) minimum threshold
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy import signal

from src.tier0_errors import BiosignalQualityError

EXPECTED_LEADS = 12
DEFAULT_SAMPLING_RATE_HZ = 500
MIN_DURATION_SECONDS = 5.0
MIN_SNR_DB = 5.0
MAX_ABS_MV = 20.0
MIN_LEAD_STD_MV = 1e-4


@dataclass(frozen=True)
class EcgInput:
    signal_mv: np.ndarray
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ


@dataclass(frozen=True)
class BiosignalReport:
    sampling_rate_hz: int
    n_leads: int
    n_samples: int
    snr_db: float
    filtered_mv: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampling_rate_hz": self.sampling_rate_hz,
            "n_leads": self.n_leads,
            "n_samples": self.n_samples,
            "snr_db": round(float(self.snr_db), 3),
        }


def load_ecg(path: Union[str, Path]) -> EcgInput:
    """Load a 12-lead ECG from .npy, .npz, or .json file."""
    p = Path(path)
    if not p.is_file():
        raise BiosignalQualityError(
            "M04", "M04_FILE_NOT_FOUND", f"ECG file not found: {p.name}"
        )
    suffix = p.suffix.lower()
    try:
        if suffix == ".npy":
            arr = np.load(str(p), allow_pickle=False)
            fs = DEFAULT_SAMPLING_RATE_HZ
        elif suffix == ".npz":
            with np.load(str(p), allow_pickle=False) as data:
                if "signal_mv" in data:
                    arr = data["signal_mv"]
                elif "ecg" in data:
                    arr = data["ecg"]
                else:
                    raise BiosignalQualityError(
                        "M04", "M04_MISSING_NPZ_KEY", "NPZ archive must contain 'signal_mv' or 'ecg'."
                    )
                fs = int(data["sampling_rate_hz"]) if "sampling_rate_hz" in data else DEFAULT_SAMPLING_RATE_HZ
        elif suffix == ".json":
            payload = json.loads(p.read_text())
            arr = np.asarray(payload["signal_mv"], dtype=np.float32)
            fs = int(payload.get("sampling_rate_hz", DEFAULT_SAMPLING_RATE_HZ))
        else:
            raise BiosignalQualityError(
                "M04", "M04_UNSUPPORTED_FORMAT", f"Unsupported ECG file extension {suffix!r}."
            )
    except BiosignalQualityError:
        raise
    except Exception as exc:
        raise BiosignalQualityError(
            "M04", "M04_CORRUPT_ECG_FILE", f"Failed to load ECG file {p.name}: {exc}"
        ) from exc

    return EcgInput(signal_mv=np.asarray(arr, dtype=np.float32), sampling_rate_hz=fs)


def bandpass_filter_ecg(
    sig: np.ndarray,
    fs: int,
    lowcut_hz: float = 0.5,
    highcut_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    nyq = 0.5 * float(fs)
    low = lowcut_hz / nyq
    high = highcut_hz / nyq
    if low <= 0 or high >= 1.0 or low >= high:
        raise BiosignalQualityError(
            "M04", "M04_INVALID_FILTER_BAND", f"Invalid filter band [{lowcut_hz}, {highcut_hz}] Hz for fs={fs} Hz."
        )
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, sig, axis=-1).astype(np.float32)


def zscore_leads(sig: np.ndarray) -> np.ndarray:
    """Per-lead zero-mean, unit-variance normalization."""
    arr = np.asarray(sig, dtype=np.float32)
    mean = np.mean(arr, axis=-1, keepdims=True)
    std = np.std(arr, axis=-1, keepdims=True)
    if np.any(std < 1e-8):
        raise BiosignalQualityError(
            "M04", "M04_ZERO_STD_AFTER_FILTER", "One or more ECG leads collapsed to zero variance."
        )
    return ((arr - mean) / std).astype(np.float32)


class BiosignalGate:
    """Validates 12-lead ECG waveform quality and applies 0.5-40 Hz bandpass filtering."""

    def __init__(
        self,
        *,
        expected_fs: int = DEFAULT_SAMPLING_RATE_HZ,
        min_duration_s: float = MIN_DURATION_SECONDS,
        min_snr_db: float = MIN_SNR_DB,
        max_abs_mv: float = MAX_ABS_MV,
    ) -> None:
        self.expected_fs = int(expected_fs)
        self.min_duration_s = float(min_duration_s)
        self.min_snr_db = float(min_snr_db)
        self.max_abs_mv = float(max_abs_mv)

    def enforce(self, ecg: Union[EcgInput, np.ndarray]) -> BiosignalReport:
        if isinstance(ecg, np.ndarray):
            inp = EcgInput(signal_mv=np.asarray(ecg, dtype=np.float32), sampling_rate_hz=self.expected_fs)
        elif isinstance(ecg, EcgInput):
            inp = ecg
        else:
            raise BiosignalQualityError("M04", "M04_INVALID_INPUT_TYPE", f"Unexpected ECG input type: {type(ecg).__name__}")

        raw = inp.signal_mv
        fs = int(inp.sampling_rate_hz)

        if fs != self.expected_fs:
            raise BiosignalQualityError(
                "M04",
                "M04_SAMPLING_RATE_MISMATCH",
                f"Sampling rate {fs} Hz does not match required {self.expected_fs} Hz.",
                {"sampling_rate_hz": fs, "expected_fs": self.expected_fs},
            )

        if raw.ndim != 2 or raw.shape[0] != EXPECTED_LEADS:
            raise BiosignalQualityError(
                "M04",
                "M04_INVALID_LEAD_COUNT",
                f"Expected shape (12, N), got {raw.shape}.",
                {"shape": list(raw.shape)},
            )

        n_samples = int(raw.shape[1])
        min_samples = int(self.min_duration_s * fs)
        if n_samples < min_samples:
            raise BiosignalQualityError(
                "M04",
                "M04_DURATION_TOO_SHORT",
                f"ECG has {n_samples} samples (< minimum {min_samples} for {self.min_duration_s}s).",
                {"n_samples": n_samples, "min_samples": min_samples},
            )

        if not np.all(np.isfinite(raw)):
            raise BiosignalQualityError(
                "M04", "M04_NON_FINITE_SAMPLES", "ECG signal contains NaN or Inf values."
            )

        if float(np.max(np.abs(raw))) > self.max_abs_mv:
            raise BiosignalQualityError(
                "M04",
                "M04_AMPLITUDE_RAIL_SATURATION",
                f"ECG amplitude exceeds +/-{self.max_abs_mv} mV rail limit.",
            )

        lead_stds = np.std(raw, axis=-1)
        dead_leads = [int(i) for i, s in enumerate(lead_stds) if s < MIN_LEAD_STD_MV]
        if dead_leads:
            raise BiosignalQualityError(
                "M04",
                "M04_LEAD_OFF_DETECTED",
                f"Flatline / lead-off detected on lead indices: {dead_leads}",
                {"leads": dead_leads},
            )

        filtered = bandpass_filter_ecg(raw, fs=fs)
        noise = raw - filtered
        sig_power = float(np.mean(filtered ** 2))
        noise_power = float(np.mean(noise ** 2)) + 1e-12
        snr_db = 10.0 * float(np.log10(max(sig_power, 1e-12) / noise_power))

        if snr_db < self.min_snr_db:
            raise BiosignalQualityError(
                "M04",
                "M04_LOW_SNR",
                f"Pooled ECG SNR {snr_db:.2f} dB is below minimum {self.min_snr_db:.2f} dB.",
                {"snr_db": snr_db, "min_snr_db": self.min_snr_db},
            )

        return BiosignalReport(
            sampling_rate_hz=fs,
            n_leads=EXPECTED_LEADS,
            n_samples=n_samples,
            snr_db=snr_db,
            filtered_mv=filtered,
        )
