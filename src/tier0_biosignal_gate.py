"""M04 - Biosignal 1-D SNR Gate (Tier 0).

Quality gate for 12-lead ECG sampled at 500 Hz (units: mV).

Per-lead checks
---------------
* ``DISCONNECTED``   - non-finite samples, or > 80 % exactly-repeated samples (lead-off /
  digital rail), or |mean| >= 10 mV (amplifier saturation).
* ``FLATLINE``       - peak-to-peak < 0.05 mV.
* ``EXTREME_BASELINE_WANDER`` - peak-to-peak of the < 0.5 Hz component > 2.0 mV.
* ``LOW_SNR``        - SNR < 6.0 dB.

SNR definition
--------------
After DC removal, the signal estimate is the zero-phase 3rd-order Butterworth
band-pass (0.5-40 Hz, ``sosfiltfilt``). The noise estimate is the residual
``x - bandpass(x)``, i.e. baseline drift (< 0.5 Hz) plus high-frequency / powerline
(> 40 Hz) energy. ``SNR_dB = 10 log10(P_signal / P_noise)``.
Limitation: broadband in-band noise (0.5-40 Hz, e.g. EMG) is, by construction, not
counted as noise by this estimator; it is a spectral gate, not a morphology audit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import signal

from src.tier0_errors import BiosignalQualityError, InputError

LOGGER = logging.getLogger(__name__)
STAGE = "M04"

SAMPLING_RATE_HZ = 500.0
N_LEADS = 12
LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
BANDPASS_HZ = (0.5, 40.0)
BUTTERWORTH_ORDER = 3
FLATLINE_PTP_MV = 0.05
MIN_SNR_DB = 6.0
EXTREME_WANDER_PTP_MV = 2.0
MIN_DURATION_S = 5.0
DISCONNECT_REPEAT_FRACTION = 0.80
RAIL_MEAN_MV = 10.0


@dataclass(frozen=True)
class LeadQuality:
    name: str
    status: str  # OK | FAIL
    reasons: List[str]
    ptp_mv: Optional[float]
    baseline_wander_ptp_mv: Optional[float]
    snr_db: Optional[float]


@dataclass(frozen=True)
class BiosignalGateResult:
    passed: bool
    snr_db: Optional[float]  # pooled over evaluable leads
    min_lead_snr_db: Optional[float]
    min_snr_threshold_db: float
    sampling_rate_hz: float
    n_samples: int
    leads: List[LeadQuality]
    failure_reasons: List[str]
    filtered_mv: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed, "snr_db": self.snr_db, "min_lead_snr_db": self.min_lead_snr_db,
            "min_snr_threshold_db": self.min_snr_threshold_db, "sampling_rate_hz": self.sampling_rate_hz,
            "n_samples": self.n_samples, "failure_reasons": list(self.failure_reasons),
            "leads": [vars(l) for l in self.leads],
        }


def load_ecg(path: Union[str, Path]) -> np.ndarray:
    """Load a 12-lead ECG from ``.npy`` or ``.csv`` (mV) as an array of shape (12, n_samples)."""
    p = Path(path)
    if not p.is_file():
        raise InputError(STAGE, "FILE_NOT_FOUND", f"ECG file does not exist: {p}")
    suffix = p.suffix.lower()
    try:
        if suffix == ".npy":
            arr = np.load(p, allow_pickle=False)
        elif suffix == ".csv":
            try:
                arr = np.loadtxt(p, delimiter=",", ndmin=2)
            except ValueError:
                arr = np.loadtxt(p, delimiter=",", ndmin=2, skiprows=1)  # header row
        else:
            raise InputError(STAGE, "UNSUPPORTED_ECG_FORMAT", f"Unsupported ECG file type '{suffix}' (use .npy or .csv).")
    except InputError:
        raise
    except Exception as exc:
        raise InputError(STAGE, "ECG_READ_FAILED", f"Failed to read ECG file: {exc!r}") from exc
    return orient_leads(arr)


def orient_leads(arr: np.ndarray) -> np.ndarray:
    """Return float64 array shaped (12, n_samples) from (12, n) or (n, 12) input."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise InputError(STAGE, "BAD_ECG_RANK", f"ECG must be 2-D, got shape {a.shape}.")
    if a.shape[0] == N_LEADS:
        return a
    if a.shape[1] == N_LEADS:
        return a.T.copy()
    raise InputError(STAGE, "BAD_LEAD_COUNT", f"ECG must have {N_LEADS} leads, got shape {a.shape}.")


class BiosignalGate:
    """M04 12-lead ECG SNR gate."""

    def __init__(
        self,
        fs: float = SAMPLING_RATE_HZ,
        min_snr_db: float = MIN_SNR_DB,
        flatline_ptp_mv: float = FLATLINE_PTP_MV,
        extreme_wander_ptp_mv: float = EXTREME_WANDER_PTP_MV,
        band_hz: tuple = BANDPASS_HZ,
        order: int = BUTTERWORTH_ORDER,
    ) -> None:
        self.fs = float(fs)
        self.min_snr_db = float(min_snr_db)
        self.flatline_ptp_mv = float(flatline_ptp_mv)
        self.extreme_wander_ptp_mv = float(extreme_wander_ptp_mv)
        self._sos_band = signal.butter(order, band_hz, btype="bandpass", fs=self.fs, output="sos")
        self._sos_wander = signal.butter(order, band_hz[0], btype="lowpass", fs=self.fs, output="sos")

    def evaluate(self, ecg_mv: np.ndarray) -> BiosignalGateResult:
        """Evaluate a (12, n) or (n, 12) array in mV. Never raises on quality failure."""
        x = orient_leads(ecg_mv)
        n = x.shape[1]
        if n < int(MIN_DURATION_S * self.fs):
            raise InputError(
                STAGE, "RECORDING_TOO_SHORT",
                f"ECG has {n} samples ({n / self.fs:.2f} s); minimum is {MIN_DURATION_S:.0f} s at {self.fs:.0f} Hz.",
            )

        leads: List[LeadQuality] = []
        failures: List[str] = []
        filtered = np.zeros_like(x)
        sig_power = noise_power = 0.0
        snrs: List[float] = []

        for i in range(N_LEADS):
            name, lead = LEAD_NAMES[i], x[i]
            reasons: List[str] = []
            ptp = wander = snr = None

            if not np.isfinite(lead).all():
                reasons.append("DISCONNECTED")
            else:
                ptp = float(np.ptp(lead))
                repeat_frac = float(np.mean(np.diff(lead) == 0.0))
                if repeat_frac > DISCONNECT_REPEAT_FRACTION or abs(float(lead.mean())) >= RAIL_MEAN_MV:
                    reasons.append("DISCONNECTED")
                if ptp < self.flatline_ptp_mv:
                    reasons.append("FLATLINE")

            if not reasons:
                centered = lead - lead.mean()
                wander = float(np.ptp(signal.sosfiltfilt(self._sos_wander, centered)))
                if wander > self.extreme_wander_ptp_mv:
                    reasons.append("EXTREME_BASELINE_WANDER")
                band = signal.sosfiltfilt(self._sos_band, centered)
                residual = centered - band
                p_sig, p_noise = float(np.var(band)), max(float(np.var(residual)), 1e-12)
                snr = 10.0 * np.log10(max(p_sig, 1e-12) / p_noise)
                filtered[i] = band
                sig_power += p_sig
                noise_power += p_noise
                snrs.append(snr)
                if snr < self.min_snr_db:
                    reasons.append("LOW_SNR")

            for r in reasons:
                failures.append(f"{name}:{r}")
            leads.append(LeadQuality(name, "FAIL" if reasons else "OK", reasons, ptp, wander,
                                     None if snr is None else float(snr)))

        pooled = float(10.0 * np.log10(max(sig_power, 1e-12) / max(noise_power, 1e-12))) if snrs else None
        return BiosignalGateResult(
            passed=not failures, snr_db=pooled, min_lead_snr_db=(float(min(snrs)) if snrs else None),
            min_snr_threshold_db=self.min_snr_db, sampling_rate_hz=self.fs, n_samples=n, leads=leads,
            failure_reasons=failures, filtered_mv=filtered if not failures else None,
        )

    def enforce(self, ecg_mv: np.ndarray) -> BiosignalGateResult:
        """Evaluate and raise :class:`BiosignalQualityError` unless every lead passes."""
        result = self.evaluate(ecg_mv)
        if not result.passed:
            raise BiosignalQualityError(
                STAGE, "SIGNAL_QUALITY_REJECTED",
                f"ECG rejected: {', '.join(result.failure_reasons)} "
                f"(pooled SNR={result.snr_db if result.snr_db is None else round(result.snr_db, 2)} dB, "
                f"threshold {self.min_snr_db} dB).",
                result.to_dict(),
            )
        return result


def zscore_leads(filtered_mv: np.ndarray) -> np.ndarray:
    """Per-lead zero-mean / unit-variance normalisation (float32, shape (12, n))."""
    mean = filtered_mv.mean(axis=1, keepdims=True)
    std = filtered_mv.std(axis=1, keepdims=True)
    return ((filtered_mv - mean) / np.maximum(std, 1e-8)).astype(np.float32)
