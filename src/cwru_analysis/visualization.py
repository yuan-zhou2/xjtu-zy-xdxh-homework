"""Visualization utilities for CWRU bearing signal analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .data import Segment, SignalRecord
from .time_frequency import WaveletConfig, compute_cwt, compute_stft

__all__ = [
    "plot_time_series",
    "plot_frequency_spectrum",
    "plot_psd",
    "plot_stft_spectrogram",
    "plot_cwt_scalogram",
    "plot_emd_imfs",
]


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei"],
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.unicode_minus": False,
        }
    )


def _configure_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_time_series(
    record: SignalRecord,
    max_seconds: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    _apply_publication_style()
    ax = ax or plt.gca()
    values = record.values
    fs = record.sample_rate
    if max_seconds is not None:
        limit = int(max_seconds * fs)
        values = values[:limit]
    t = np.arange(values.size) / fs
    ax.plot(t, values, color="#1f77b4", linewidth=1.0)
    _configure_axes(
        ax,
        title or f"时间域信号：{record.sensor.value} ({record.path.stem})",
        xlabel="时间 / s",
        ylabel="幅值",
    )
    return ax


def plot_frequency_spectrum(
    segment: Segment,
    n_fft: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    _apply_publication_style()
    ax = ax or plt.gca()
    x = segment.samples
    n = n_fft or int(2 ** np.ceil(np.log2(x.size)))
    window = np.hanning(x.size)
    spectrum = np.fft.rfft(window * x, n=n)
    freqs = np.fft.rfftfreq(n, d=1 / segment.record.sample_rate)
    magnitude = np.abs(spectrum)
    ax.plot(freqs, magnitude, color="#ff7f0e")
    _configure_axes(
        ax,
        title or "幅度谱",
        xlabel="频率 / Hz",
        ylabel="幅度",
    )
    ax.set_xlim(0, segment.record.sample_rate / 2)
    return ax


def plot_psd(
    segment: Segment,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    method: str = "welch",
) -> plt.Axes:
    _apply_publication_style()
    ax = ax or plt.gca()
    try:
        from scipy import signal as sp_signal
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for PSD plotting") from exc

    fs = segment.record.sample_rate
    if method == "periodogram":
        freqs, psd = sp_signal.periodogram(segment.samples, fs=fs, window="hann", scaling="density")
    else:
        freqs, psd = sp_signal.welch(segment.samples, fs=fs, window="hann", nperseg=min(1024, segment.samples.size))
    ax.semilogy(freqs, psd, color="#2ca02c")
    _configure_axes(
        ax,
        title or "功率谱密度",
        xlabel="频率 / Hz",
        ylabel="PSD",
    )
    ax.set_xlim(0, fs / 2)
    return ax


def plot_stft_spectrogram(
    segment: Segment,
    ax: Optional[plt.Axes] = None,
    config: Optional[Dict[str, int]] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    _apply_publication_style()
    ax = ax or plt.gca()
    config = config or {}
    f, t, z = compute_stft(segment, **config)
    magnitude = np.abs(z)
    mesh = ax.pcolormesh(t, f, magnitude, shading="auto", cmap="viridis")
    plt.colorbar(mesh, ax=ax, label="幅值")
    _configure_axes(
        ax,
        title or "短时傅里叶谱图",
        xlabel="时间 / s",
        ylabel="频率 / Hz",
    )
    ax.set_ylim(0, segment.record.sample_rate / 2)
    return ax


def plot_cwt_scalogram(
    segment: Segment,
    ax: Optional[plt.Axes] = None,
    config: Optional[WaveletConfig] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    _apply_publication_style()
    ax = ax or plt.gca()
    config = config or WaveletConfig()
    coefficients, frequencies, _ = compute_cwt(segment, config)
    t = np.arange(segment.samples.size) / segment.record.sample_rate
    power = np.abs(coefficients)
    mesh = ax.pcolormesh(t, frequencies, power, shading="auto", cmap="inferno")
    plt.colorbar(mesh, ax=ax, label="幅值")
    _configure_axes(
        ax,
        title or "连续小波变换尺度图",
        xlabel="时间 / s",
        ylabel="频率 / Hz",
    )
    ax.set_ylim(frequencies.min(), frequencies.max())
    ax.set_yscale("log")
    return ax


def plot_emd_imfs(
    imfs: Dict[str, np.ndarray],
    sample_rate: float,
    fig: Optional[plt.Figure] = None,
    title: str = "EMD Intrinsic Mode Functions",
) -> plt.Figure:
    _apply_publication_style()
    if fig is None:
        fig, axes = plt.subplots(len(imfs), 1, figsize=(12, 2 * len(imfs)), sharex=True)
    else:
        axes = fig.subplots(len(imfs), 1, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (name, imf) in zip(axes, imfs.items()):
        t = np.arange(imf.size) / sample_rate
        ax.plot(t, imf, linewidth=0.8)
        _configure_axes(ax, f"{title} - {name}", xlabel="时间 / s", ylabel="幅值")
    axes[-1].set_xlabel("时间 / s")
    fig.tight_layout()
    return fig
