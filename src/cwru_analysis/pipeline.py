"""End-to-end analysis pipelines for the CWRU dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import Segment, SignalRecord, generate_segments, load_catalog
from .features import aggregate_feature_table
from .metadata import FaultType, SensorLocation

__all__ = [
    "SegmentConfig",
    "PipelineResult",
    "summarize_by_fault",
    "summarize_by_fault_and_load",
    "run_feature_pipeline",
]


@dataclass(frozen=True)
class SegmentConfig:
    window_size: int = 2048
    step_size: int = 1024
    normalize: bool = True
    centered: bool = True


@dataclass
class PipelineResult:
    records: List[SignalRecord]
    segments: List[Segment]
    feature_table: pd.DataFrame
    summary_by_fault: pd.DataFrame
    summary_by_fault_load: pd.DataFrame


def run_feature_pipeline(
    root: Path,
    fault_types: Optional[Iterable[FaultType]] = None,
    sensor: SensorLocation = SensorLocation.DRIVE_END,
    segment_config: SegmentConfig = SegmentConfig(),
    max_records: Optional[int] = None,
    export_dir: Optional[Path] = None,
) -> PipelineResult:
    fault_selector: Optional[Sequence[FaultType]] = list(fault_types) if fault_types else None
    records: List[SignalRecord] = []
    segments: List[Segment] = []
    for idx, record in enumerate(
        load_catalog(root, fault_types=fault_selector, sensor=sensor)
    ):
        if max_records is not None and idx >= max_records:
            break
        records.append(record)
        for segment in generate_segments(
            record,
            window_size=segment_config.window_size,
            step_size=segment_config.step_size,
            normalize=segment_config.normalize,
            centered=segment_config.centered,
        ):
            segments.append(segment)
    feature_table = aggregate_feature_table(segments)
    summary_fault = summarize_by_fault(feature_table)
    summary_fault_load = summarize_by_fault_and_load(feature_table)
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
        feature_table.to_csv(export_dir / "feature_table.csv", index=False)
        summary_fault.to_csv(export_dir / "summary_by_fault.csv")
        summary_fault_load.to_csv(export_dir / "summary_by_fault_load.csv")
    return PipelineResult(
        records=records,
        segments=segments,
        feature_table=feature_table,
        summary_by_fault=summary_fault,
        summary_by_fault_load=summary_fault_load,
    )


def summarize_by_fault(feature_table: pd.DataFrame) -> pd.DataFrame:
    if feature_table.empty:
        return pd.DataFrame()
    numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
    return (
        feature_table.groupby("fault_type")[numeric_cols]
        .agg(["mean", "std"])
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )


def summarize_by_fault_and_load(feature_table: pd.DataFrame) -> pd.DataFrame:
    if feature_table.empty:
        return pd.DataFrame()
    numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
    grouped = feature_table.groupby(["fault_type", "load"])[numeric_cols].agg(["mean", "std"])
    return grouped.swaplevel(axis=1).sort_index(axis=1)
