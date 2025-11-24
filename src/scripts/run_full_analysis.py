"""Command-line entry point for running the comprehensive CWRU analysis pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if src_dir not in map(Path, map(Path, sys.path)):
        sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

from cwru_analysis.metadata import FaultType, SensorLocation
from cwru_analysis.pipeline import SegmentConfig, run_feature_pipeline


DEFAULT_ROOT = Path("12k Drive End Bearing Fault Data")
DEFAULT_EXPORT = Path("outputs/12k_drive_end")


FAULT_TYPE_CHOICES = [ft.value for ft in FaultType]
SENSOR_CHOICES = [sensor.value for sensor in SensorLocation]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CWRU signal analysis workflow")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Path to dataset root directory (default: 12k Drive End subset)",
    )
    parser.add_argument(
        "--fault-types",
        nargs="*",
        default=[],
        choices=FAULT_TYPE_CHOICES,
        help="Subset of fault types to include (default: all)",
    )
    parser.add_argument(
        "--sensor",
        default=SensorLocation.DRIVE_END.value,
        choices=SENSOR_CHOICES,
        help="Sensor location to analyze",
    )
    parser.add_argument("--window", type=int, default=2048, help="Segment window size (samples)")
    parser.add_argument("--step", type=int, default=1024, help="Segment hop size (samples)")
    parser.add_argument("--max-records", type=int, default=None, help="Limit number of records for faster debugging")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Directory to write CSV summaries (default auto-selected for 12k preset)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fault_types = [FaultType(ft) for ft in args.fault_types] if args.fault_types else None
    sensor = SensorLocation(args.sensor)
    segment_config = SegmentConfig(window_size=args.window, step_size=args.step)

    export_dir = args.export if args.export is not None else (DEFAULT_EXPORT if args.root == DEFAULT_ROOT else None)

    result = run_feature_pipeline(
        root=args.root,
        fault_types=fault_types,
        sensor=sensor,
        segment_config=segment_config,
        max_records=args.max_records,
        export_dir=export_dir,
    )

    print("Records loaded:", len(result.records))
    print("Segments generated:", len(result.segments))
    print("Feature table shape:", result.feature_table.shape)
    if not result.summary_by_fault.empty:
        print("\nSummary by fault type:")
        print(result.summary_by_fault.head())
    if not result.summary_by_fault_load.empty:
        print("\nSummary by fault type and load:")
        print(result.summary_by_fault_load.head())


if __name__ == "__main__":
    main()
