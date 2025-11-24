"""Feature extraction utilities across time, frequency, and time-frequency domains."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd

from .data import Segment

__all__ = [
    "FeatureSet",
    "compute_time_features",
    "compute_frequency_features",
    "compute_time_frequency_features",
    "aggregate_feature_table",
]


@dataclass(frozen=True)
class FeatureSet:
    """Collection of feature dictionaries indexed by domain."""

    time: Mapping[str, float]
    frequency: Mapping[str, float]
    time_frequency: Mapping[str, float]

    def to_series(self) -> pd.Series:
        merged: Dict[str, float] = {}
        merged.update({f"time_{k}": v for k, v in self.time.items()})
        merged.update({f"freq_{k}": v for k, v in self.frequency.items()})
        merged.update({f"tf_{k}": v for k, v in self.time_frequency.items()})
        return pd.Series(merged)


def compute_time_features(segment: Segment) -> Dict[str, float]:
    """Compute descriptive statistics and classical vibration indicators."""

    x = segment.samples
    mean = np.mean(x)
    centered = x - mean
    abs_x = np.abs(x)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(abs_x)

    energy = np.sum(x**2)
    variance = np.var(x)
    std = np.sqrt(variance)
    skewness = _moment(centered, 3) / (std**3 + 1e-12)
    kurtosis = _moment(centered, 4) / (variance**2 + 1e-12)

    crest_factor = peak / (rms + 1e-12)
    clearance_factor = peak / (np.mean(np.sqrt(abs_x)) ** 2 + 1e-12)
    impulse_factor = peak / (np.mean(abs_x) + 1e-12)
    shape_factor = rms / (np.mean(abs_x) + 1e-12)
    margin_factor = peak / (np.mean(abs_x**0.5) ** 2 + 1e-12)

    zero_crossings = np.sum(np.diff(np.signbit(x)) != 0)

    # Hjorth parameters
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    activity = np.var(x)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-12))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-12)) / (mobility + 1e-12)

    return {
        "mean": mean,
        "std": std,
        "variance": variance,
        "rms": rms,
        "energy": energy,
        "peak": peak,
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "crest_factor": crest_factor,
        "clearance_factor": clearance_factor,
        "impulse_factor": impulse_factor,
        "shape_factor": shape_factor,
        "margin_factor": margin_factor,
        "zero_crossings": float(zero_crossings),
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def compute_frequency_features(
    segment: Segment,
    n_fft: Optional[int] = None,
    window: str = "hann",
    psd_method: str = "welch",
) -> Dict[str, float]:
    """Compute spectral metrics from the FFT/PSD."""

    x = segment.samples
    fs = segment.record.sample_rate
    length = x.size
    n = int(2 ** np.ceil(np.log2(length))) if n_fft is None else n_fft

    try:
        from scipy import signal as sp_signal
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for frequency features") from exc

    if psd_method == "periodogram":
        freqs, psd = sp_signal.periodogram(x, fs=fs, window=window, scaling="density")
    else:
        freqs, psd = sp_signal.welch(x, fs=fs, window=window, nperseg=min(1024, length), scaling="density")

    psd = psd.astype(float)
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spectral_mean = np.sum(freqs * psd_norm)
    spectral_var = np.sum(((freqs - spectral_mean) ** 2) * psd_norm)

    idx_max = np.argmax(psd)
    peak_freq = freqs[idx_max]
    spectral_skewness = np.sum(((freqs - spectral_mean) ** 3) * psd_norm) / (spectral_var ** 1.5 + 1e-12)
    spectral_kurtosis = np.sum(((freqs - spectral_mean) ** 4) * psd_norm) / (spectral_var ** 2 + 1e-12)

    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    cumulative = np.cumsum(psd_norm)
    roll_off_95 = freqs[np.searchsorted(cumulative, 0.95)]

    bandwidth = np.sqrt(spectral_var)
    flatness = np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)

    # Envelope spectrum via Hilbert transform
    from .preprocessing import hilbert_envelope

    envelope = hilbert_envelope(x)
    env_freqs, env_psd = sp_signal.periodogram(envelope, fs=fs, window=window, scaling="density")
    env_peak_freq = env_freqs[np.argmax(env_psd)]

    return {
        "spectral_mean": spectral_mean,
        "spectral_var": spectral_var,
        "spectral_std": np.sqrt(spectral_var),
        "spectral_skewness": spectral_skewness,
        "spectral_kurtosis": spectral_kurtosis,
        "spectral_entropy": spectral_entropy,
        "spectral_flatness": flatness,
        "bandwidth": bandwidth,
        "peak_frequency": peak_freq,
        "roll_off_95": roll_off_95,
        "envelope_peak_frequency": env_peak_freq,
    }


def compute_time_frequency_features(
    segment: Segment,
    wavelet: str = "morl",
    scales: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Derive features from STFT, wavelet scalograms, and EMD where available."""

    x = segment.samples
    fs = segment.record.sample_rate
    try:
        from scipy import signal as sp_signal
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for STFT features") from exc

    f, t, z = sp_signal.stft(x, fs=fs, window="hann", nperseg=min(1024, x.size))
    magnitude = np.abs(z)
    stft_energy = float(np.sum(magnitude**2))
    stft_entropy = -float(np.sum(_normalize(magnitude) * np.log(_normalize(magnitude) + 1e-12)))

    import pywt

    if scales is None:
        max_scale = min(128, max(16, x.size // 4))
        scales = np.arange(1, max_scale)
    coefficients, freqs = pywt.cwt(x, scales=scales, wavelet=wavelet, sampling_period=1 / fs)
    scalogram = np.abs(coefficients)
    scale_energy = np.sum(scalogram**2, axis=1)
    dominant_scale_idx = int(np.argmax(scale_energy))
    dominant_frequency = freqs[dominant_scale_idx]
    scale_entropy = -np.sum(_normalize(scale_energy) * np.log(_normalize(scale_energy) + 1e-12))

    emd_features = _emd_metrics(x, fs)

    return {
        "stft_energy": stft_energy,
        "stft_entropy": stft_entropy,
        "cwt_dominant_frequency": float(dominant_frequency),
        "cwt_scale_entropy": float(scale_entropy),
        **emd_features,
    }


def aggregate_feature_table(segments: Iterable[Segment]) -> pd.DataFrame:
    """Compute a concatenated feature table for many segments."""

    frames: List[pd.Series] = []
    for idx, segment in enumerate(segments):
        feature_set = FeatureSet(
            time=compute_time_features(segment),
            frequency=compute_frequency_features(segment),
            time_frequency=compute_time_frequency_features(segment),
        )
        series = feature_set.to_series()
        series["segment_index"] = idx
        series["rpm"] = segment.record.rpm if segment.record.rpm is not None else np.nan
        series["sensor"] = segment.record.sensor.value
        series["fault_type"] = (
            segment.record.fault_type.value
            if segment.record.fault_type is not None
            else segment.record.path.parent.parent.name
        )
        series["load"] = (
            segment.record.load.value if segment.record.load is not None else "unknown"
        )
        series["fault_size"] = (
            segment.record.fault_size if segment.record.fault_size is not None else np.nan
        )
        frames.append(series)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Helpers

def _moment(x: np.ndarray, order: int) -> float:
    return float(np.mean(x**order))


def _normalize(values: np.ndarray) -> np.ndarray:
    total = np.sum(values)
    if total <= 0:
        return np.zeros_like(values)
    return values / total


def _emd_metrics(signal: np.ndarray, fs: float) -> Dict[str, float]:
    try:
        from PyEMD import EMD
    except ImportError:
        return {
            "emd_imf_count": 0.0,
            "emd_energy_ratio": 0.0,
            "emd_hilbert_energy": 0.0,
        }

    emd = EMD()
    imfs = emd(signal)
    if imfs.size == 0:
        return {
            "emd_imf_count": 0.0,
            "emd_energy_ratio": 0.0,
            "emd_hilbert_energy": 0.0,
        }
    energies = np.sum(imfs**2, axis=1)
    total_energy = np.sum(energies) + 1e-12
    top3 = np.sort(energies)[-3:]
    energy_ratio = float(np.sum(top3) / total_energy)

    try:
        from PyEMD import HilbertSpectra
    except ImportError:
        hilbert_energy = float(total_energy)
    else:
        hs = HilbertSpectra(signal, delta_t=1 / fs)
        amplitude, *_ = hs.get_hilbert_envelope(imfs)
        hilbert_energy = float(np.sum(amplitude**2))

    return {
        "emd_imf_count": float(imfs.shape[0]),
        "emd_energy_ratio": energy_ratio,
        "emd_hilbert_energy": hilbert_energy,
    }
