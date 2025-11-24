"""Dataset metadata utilities for the CWRU bearing fault corpus."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

__all__ = [
    "SensorLocation",
    "FaultType",
    "LoadCondition",
    "Acquisition",
    "CWRUIndex",
]


class SensorLocation(str, Enum):
    """Available sensor mounting positions in the CWRU dataset."""

    DRIVE_END = "DE"
    FAN_END = "FE"
    BASE_ACCEL = "BA"
    MOTOR_CURRENT = "MC"


class FaultType(str, Enum):
    """Enumerated bearing health conditions."""

    NORMAL = "normal"
    BALL = "ball"
    INNER_RACE = "inner_race"
    OUTER_RACE = "outer_race"


class LoadCondition(str, Enum):
    """Discrete load conditions present in the dataset (in horsepower)."""

    HP_0 = "0hp"
    HP_1 = "1hp"
    HP_2 = "2hp"
    HP_3 = "3hp"

    @classmethod
    def from_rpm(cls, rpm: float) -> "LoadCondition":
        """Map RPM range to the closest nominal load condition."""

        if rpm < 1200:
            return cls.HP_0
        if rpm < 1500:
            return cls.HP_1
        if rpm < 1750:
            return cls.HP_2
        return cls.HP_3


@dataclass(frozen=True)
class Acquisition:
    """Structured description of a single .mat acquisition file."""

    fault_type: FaultType
    load: LoadCondition
    fault_size: Optional[float]
    sensor_path: Path
    rpm: Optional[float]

    @property
    def identifier(self) -> str:
        """Unique identifier string for report tables and plots."""

        size_fragment = f"_{self.fault_size:.3f}" if self.fault_size else ""
        return f"{self.fault_type.value}{size_fragment}_{self.load.value}"


class CWRUIndex:
    """Helper class that builds a catalog of CWRU files from a root folder."""

    DEFAULT_MAT_KEYS = {
        SensorLocation.DRIVE_END: "DE_time",
        SensorLocation.FAN_END: "FE_time",
        SensorLocation.BASE_ACCEL: "BA_time",
        SensorLocation.MOTOR_CURRENT: "MC_time",
    }

    def __init__(self, root: Path) -> None:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        self.root = root

    def iter_acquisitions(
        self,
        fault_types: Optional[Sequence[FaultType]] = None,
    ) -> Iterator[Acquisition]:
        """Yield acquisitions for the requested fault subset."""

        selectors = set(fault_types or list(FaultType))
        for condition_dir in self._candidate_dirs():
            fault_type = self._infer_fault_type(condition_dir)
            if fault_type not in selectors:
                continue
            for mat_file in sorted(condition_dir.glob("*.mat")):
                load = self._infer_load(mat_file.name)
                fault_size = self._infer_fault_size(mat_file.name)
                rpm = self._extract_rpm(mat_file)
                yield Acquisition(
                    fault_type=fault_type,
                    load=load,
                    fault_size=fault_size,
                    sensor_path=mat_file,
                    rpm=rpm,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    def _candidate_dirs(self) -> Iterable[Path]:
        root_has_mat = False
        for child in self.root.iterdir():
            if child.is_dir():
                if any(grandchild.suffix == ".mat" for grandchild in child.iterdir()):
                    yield child
                    continue
                for nested in child.iterdir():
                    if nested.is_dir() and any(file.suffix == ".mat" for file in nested.iterdir()):
                        yield nested
            elif child.suffix == ".mat":
                root_has_mat = True
        if root_has_mat:
            yield self.root

    @staticmethod
    def _infer_fault_type(path: Path) -> FaultType:
        candidates = [path.name.lower()]
        parent = path.parent
        # Inspect up to two levels of parents for descriptive folder names
        for _ in range(3):
            if parent is None:
                break
            candidates.append(parent.name.lower())
            parent = parent.parent if parent != parent.parent else None

        for name in candidates:
            if "ball" in name:
                return FaultType.BALL
            if "inner" in name:
                return FaultType.INNER_RACE
            if "outer" in name:
                return FaultType.OUTER_RACE
            if "normal" in name:
                return FaultType.NORMAL
        return FaultType.NORMAL

    @staticmethod
    def _infer_fault_size(filename: str) -> Optional[float]:
        for token in filename.replace("-", "_").split("_"):
            if token.endswith("mm"):
                try:
                    return float(token.rstrip("mm"))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _infer_load(filename: str) -> LoadCondition:
        lowered = filename.lower()
        if "1" in lowered:
            return LoadCondition.HP_1
        if "2" in lowered:
            return LoadCondition.HP_2
        if "3" in lowered:
            return LoadCondition.HP_3
        return LoadCondition.HP_0

    @staticmethod
    def _extract_rpm(mat_file: Path) -> Optional[float]:
        try:
            import scipy.io as sio  # type: ignore

            data = sio.loadmat(mat_file, squeeze_me=True)
        except Exception:
            return None
        for key in data:
            if key.lower().endswith("rpm"):
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    continue
        return None

    # ------------------------------------------------------------------
    def describe_counts(self) -> Dict[FaultType, int]:
        """Summarize number of acquisitions per fault type."""

        counts: Dict[FaultType, int] = {ft: 0 for ft in FaultType}
        for acquisition in self.iter_acquisitions():
            counts[acquisition.fault_type] += 1
        return counts

    def available_sensors(self, mat_file: Path) -> List[SensorLocation]:
        """Return the sensors present within a given .mat file."""

        try:
            import scipy.io as sio  # type: ignore

            data = sio.loadmat(mat_file)
        except Exception:  # pragma: no cover - IO errors handled upstream
            return []
        sensors: List[SensorLocation] = []
        for sensor, suffix in self.DEFAULT_MAT_KEYS.items():
            if any(key.lower().endswith(suffix.lower()) for key in data):
                sensors.append(sensor)
        return sensors
