"""Signal preprocessing utilities for CWRU analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

__all__ = [
    "FilterConfig",
    "remove_dc",
    "detrend_signal",
    "bandpass",
    "lowpass",
    "highpass",
    "apply_filter",
    "rolling_statistic",
    "hilbert_envelope",
    "resample_signal",
]


@dataclass(frozen=True)
class FilterConfig:
    """Finite impulse response/IIR filter definition."""

    btype: str
    cutoff: Tuple[float, float] | float
    order: int = 4
    design: str = "butter"

    def normalized(self, fs: float) -> Tuple[float, float] | float:
        if isinstance(self.cutoff, Iterable):
            lo, hi = self.cutoff
            return (lo / (0.5 * fs), hi / (0.5 * fs))
        return self.cutoff / (0.5 * fs)


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """Subtract the mean value."""

    return signal - np.mean(signal)


def detrend_signal(signal: np.ndarray, type: str = "linear") -> np.ndarray:
    """Remove linear trend using :func:`scipy.signal.detrend`."""

    try:
        from scipy import signal as sp_signal
    except ImportError as exc:  # pragma: no cover - SciPy availability
        raise RuntimeError("scipy is required for detrending") from exc
    return sp_signal.detrend(signal, type=type)


def _design_filter(config: FilterConfig, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy import signal as sp_signal
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for filter design") from exc
    normalized = config.normalized(fs)
    if config.design == "butter":
        return sp_signal.butter(config.order, normalized, btype=config.btype)
    if config.design == "cheby1":
        return sp_signal.cheby1(config.order, 0.5, normalized, btype=config.btype)
    if config.design == "ellip":
        return sp_signal.ellip(config.order, 0.5, 40.0, normalized, btype=config.btype)
    raise ValueError(f"Unsupported design {config.design}")


def apply_filter(signal: np.ndarray, fs: float, config: FilterConfig) -> np.ndarray:
    """Apply zero-phase forward-backward filtering."""

    try:
        from scipy import signal as sp_signal
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for filtering") from exc
    b, a = _design_filter(config, fs)
    return sp_signal.filtfilt(b, a, signal)


def bandpass(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    config = FilterConfig(btype="bandpass", cutoff=(low, high), order=order)
    return apply_filter(signal, fs, config)


def lowpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    config = FilterConfig(btype="lowpass", cutoff=cutoff, order=order)
    return apply_filter(signal, fs, config)


def highpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    config = FilterConfig(btype="highpass", cutoff=cutoff, order=order)
    return apply_filter(signal, fs, config)


def rolling_statistic(signal: np.ndarray, window: int, statistic: str = "rms") -> np.ndarray:
    """Compute rolling statistic (mean, std, rms)."""

    if window <= 0:
        raise ValueError("window must be positive")
    pad = window // 2
    padded = np.pad(signal, pad, mode="reflect")
    cumsum = np.cumsum(padded, dtype=float)
    cumsum_sq = np.cumsum(padded**2, dtype=float)
    mean = (cumsum[window:] - cumsum[:-window]) / window
    if statistic == "mean":
        return mean
    var = (cumsum_sq[window:] - cumsum_sq[:-window]) / window - mean**2
    var = np.maximum(var, 0.0)
    if statistic == "std":
        return np.sqrt(var)
    if statistic == "rms":
        return np.sqrt(mean**2 + var)
    raise ValueError(f"Unsupported statistic {statistic}")


def hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """Compute analytic signal envelope via Hilbert transform."""

    try:
        from scipy import signal as sp_signal
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for Hilbert transform") from exc
    analytic = sp_signal.hilbert(signal)
    return np.abs(analytic)


def resample_signal(signal: np.ndarray, fs: float, target_fs: float) -> Tuple[np.ndarray, float]:
    """Resample signal using Fourier-method resampling."""

    if fs == target_fs:
        return signal, fs
    try:
        from scipy import signal as sp_signal
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for resampling") from exc
    duration = signal.size / fs
    target_length = int(round(duration * target_fs))
    resampled = sp_signal.resample(signal, target_length)
    return resampled, target_fs
