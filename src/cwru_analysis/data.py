"""Data ingestion utilities for the CWRU dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from .metadata import CWRUIndex, FaultType, LoadCondition, SensorLocation

__all__ = [
    "SignalRecord",
    "Segment",
    "load_signal",
    "generate_segments",
]


@dataclass(frozen=True)
class SignalRecord:
    """Container for a full-length acquisition from a specific sensor."""

    path: Path
    sensor: SensorLocation
    sample_rate: float
    rpm: Optional[float]
    values: np.ndarray
    fault_type: Optional[FaultType] = None
    load: Optional[LoadCondition] = None
    fault_size: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.values.size / self.sample_rate


@dataclass(frozen=True)
class Segment:
    """Short-time segment extracted from a :class:`SignalRecord`."""

    record: SignalRecord
    samples: np.ndarray
    start_index: int

    @property
    def time_axis(self) -> np.ndarray:
        step = 1.0 / self.record.sample_rate
        return np.arange(self.samples.size) * step

    @property
    def absolute_time_axis(self) -> np.ndarray:
        step = 1.0 / self.record.sample_rate
        start_time = self.start_index * step
        return start_time + np.arange(self.samples.size) * step


_SAMPLE_RATE_HINTS: Dict[str, float] = {
    "12k": 12_000.0,
    "48k": 48_000.0,
    "96k": 96_000.0,
}

_SENSOR_SUFFIX_OVERRIDES: Dict[SensorLocation, Tuple[str, ...]] = {
    SensorLocation.DRIVE_END: ("DE_time", "DE"),
    SensorLocation.FAN_END: ("FE_time", "FE"),
    SensorLocation.BASE_ACCEL: ("BA_time", "BA"),
    SensorLocation.MOTOR_CURRENT: ("MC_time", "MC"),
}


def load_signal(
    mat_path: Path,
    sensor: SensorLocation = SensorLocation.DRIVE_END,
    squeeze: bool = True,
    fault_type: Optional[FaultType] = None,
    load: Optional[LoadCondition] = None,
    fault_size: Optional[float] = None,
) -> SignalRecord:
    """Load a single sensor signal from a MAT file."""

    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    try:
        import scipy.io as sio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy is required to load MATLAB files") from exc

    raw = sio.loadmat(mat_path, squeeze_me=squeeze)
    suffixes = _SENSOR_SUFFIX_OVERRIDES[sensor]
    key = _select_key(raw, suffixes)
    values = np.asarray(raw[key], dtype=float).reshape(-1)

    rpm = _extract_rpm(raw)
    sample_rate = _infer_sample_rate(mat_path)
    return SignalRecord(
        path=mat_path,
        sensor=sensor,
        sample_rate=sample_rate,
        rpm=rpm,
        values=values,
        fault_type=fault_type,
        load=load,
        fault_size=fault_size,
    )


def generate_segments(
    record: SignalRecord,
    window_size: int,
    step_size: Optional[int] = None,
    normalize: bool = False,
    centered: bool = False,
) -> Iterator[Segment]:
    """Yield overlapping segments from a :class:`SignalRecord`."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    step = step_size or window_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    values = record.values
    if centered:
        values = values - values.mean()
    length = values.size
    for start in range(0, length - window_size + 1, step):
        window = values[start : start + window_size]
        if normalize:
            std = np.std(window)
            if std > 0:
                window = (window - np.mean(window)) / std
            else:
                window = window - np.mean(window)
        yield Segment(record=record, samples=window, start_index=start)


def _select_key(raw: Dict[str, np.ndarray], suffixes: Tuple[str, ...]) -> str:
    lowered = {key.lower(): key for key in raw.keys()}
    for suffix in suffixes:
        for key_lower, original in lowered.items():
            if key_lower.endswith(suffix.lower()):
                return original
    candidates = ", ".join(sorted(raw))
    raise KeyError(f"No matching sensor key for suffixes {suffixes}: {candidates}")


def _extract_rpm(raw: Dict[str, np.ndarray]) -> Optional[float]:
    for key, value in raw.items():
        if key.lower().endswith("rpm"):
            try:
                return float(np.asarray(value).squeeze())
            except (TypeError, ValueError):
                continue
    return None


def _infer_sample_rate(mat_path: Path) -> float:
    """Infer sample rate from directory name tokens."""

    parts = [p.lower() for p in mat_path.parts]
    for token in parts:
        for hint, rate in _SAMPLE_RATE_HINTS.items():
            if hint in token:
                return rate
    # Fallback to default 12 kHz per CWRU documentation
    return 12_000.0


def load_catalog(
    root: Path,
    fault_types: Optional[Sequence[FaultType]] = None,
    sensor: SensorLocation = SensorLocation.DRIVE_END,
    loads: Optional[Sequence[LoadCondition]] = None,
) -> Iterable[SignalRecord]:
    """Convenience generator to load multiple records from the dataset tree."""

    index = CWRUIndex(root)
    targets = list(fault_types) if fault_types else None
    load_set = set(loads) if loads else None
    for acquisition in index.iter_acquisitions(targets):
        if load_set and acquisition.load not in load_set:
            continue
        if not acquisition.sensor_path.exists():
            continue
        try:
            yield load_signal(
                acquisition.sensor_path,
                sensor=sensor,
                fault_type=acquisition.fault_type,
                load=acquisition.load,
                fault_size=acquisition.fault_size,
            )
        except KeyError:
            continue
