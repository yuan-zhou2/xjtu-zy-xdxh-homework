"""Time-frequency domain analysis utilities (STFT, wavelets, lifting, EMD)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .data import Segment

__all__ = [
    "STFTConfig",
    "WaveletConfig",
    "compute_stft",
    "compute_cwt",
    "compute_lifting_wavelet",
    "compute_emd",
    "plot_stft_figure",
    "plot_cwt_figure",
    "plot_emd_figure",
]


@dataclass(frozen=True)
class STFTConfig:
    nperseg: int = 1024
    noverlap: Optional[int] = None
    window: str = "hann"


@dataclass(frozen=True)
class WaveletConfig:
    wavelet: str = "morl"
    min_scale: int = 1
    max_scale: int = 128


def compute_stft(segment: Segment, config: STFTConfig = STFTConfig()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy import signal as sp_signal
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for STFT computations") from exc

    fs = segment.record.sample_rate
    noverlap = config.noverlap if config.noverlap is not None else config.nperseg // 2
    f, t, z = sp_signal.stft(
        segment.samples,
        fs=fs,
        window=config.window,
        nperseg=min(config.nperseg, segment.samples.size),
        noverlap=noverlap,
        boundary=None,
    )
    return f, t, z


def compute_cwt(segment: Segment, config: WaveletConfig = WaveletConfig()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pywt

    fs = segment.record.sample_rate
    scales = np.arange(config.min_scale, min(config.max_scale, segment.samples.size // 2))
    coefficients, frequencies = pywt.cwt(segment.samples, scales=scales, wavelet=config.wavelet, sampling_period=1 / fs)
    return coefficients, frequencies, scales


def compute_lifting_wavelet(signal: np.ndarray, wavelet: str = "db4", level: Optional[int] = None) -> Dict[str, np.ndarray]:
    import pywt

    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level, mode="periodization")
    reconstructions = {}
    for idx, coeff in enumerate(coeffs):
        key = "a" if idx == 0 else f"d{idx}"
        reconstructions[key] = pywt.waverec([coeff if i == idx else np.zeros_like(c) for i, c in enumerate(coeffs)], wavelet=wavelet, mode="periodization")
    return reconstructions


def compute_emd(signal: np.ndarray, max_imf: Optional[int] = None) -> Dict[str, np.ndarray]:
    try:
        from PyEMD import EMD
    except ImportError:
        raise RuntimeError("PyEMD package is required for EMD analysis")

    emd = EMD()
    imfs = emd.emd(signal, max_imf=max_imf)
    return {f"imf_{idx}": imf for idx, imf in enumerate(imfs, start=1)}


def _apply_publication_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.unicode_minus": False,
    })


def plot_stft_figure(segment: Segment, config: STFTConfig = STFTConfig()) -> "plt.Figure":
    import matplotlib.pyplot as plt

    _apply_publication_style()
    f, t, z = compute_stft(segment, config)
    magnitude = np.abs(z)
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(t, f, magnitude, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="幅值")
    ax.set_title("短时傅里叶变换谱图")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("频率 / Hz")
    ax.set_ylim(0, segment.record.sample_rate / 2)
    fig.tight_layout()
    return fig


def plot_cwt_figure(segment: Segment, config: WaveletConfig = WaveletConfig()) -> "plt.Figure":
    import matplotlib.pyplot as plt

    _apply_publication_style()
    coefficients, frequencies, _ = compute_cwt(segment, config)
    t = np.arange(segment.samples.size) / segment.record.sample_rate
    power = np.abs(coefficients)
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(t, frequencies, power, shading="auto", cmap="inferno")
    fig.colorbar(mesh, ax=ax, label="幅值")
    ax.set_title("连续小波变换尺度图")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("频率 / Hz")
    ax.set_ylim(frequencies.min(), frequencies.max())
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_emd_figure(imfs: Dict[str, np.ndarray], sample_rate: float) -> "plt.Figure":
    import matplotlib.pyplot as plt

    _apply_publication_style()
    count = len(imfs)
    fig, axes = plt.subplots(count, 1, figsize=(12, 2 * count), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (name, imf) in zip(axes, imfs.items()):
        t = np.arange(imf.size) / sample_rate
        ax.plot(t, imf, linewidth=1.0)
        ax.set_title(f"IMF {name}")
        ax.set_ylabel("幅值")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("时间 / s")
    fig.tight_layout()
    return fig
