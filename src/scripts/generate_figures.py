"""Generate publication-ready visualizations for the 12k Drive End dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

from cwru_analysis.data import Segment, load_signal
from cwru_analysis.metadata import FaultType, LoadCondition, SensorLocation
from cwru_analysis.preprocessing import remove_dc
from cwru_analysis.time_frequency import (
    STFTConfig,
    WaveletConfig,
    compute_emd,
    plot_cwt_figure,
    plot_emd_figure,
    plot_stft_figure,
)
from cwru_analysis.visualization import (
    plot_frequency_spectrum,
    plot_psd,
    plot_time_series,
)

DEFAULT_ROOT = Path("12k Drive End Bearing Fault Data")
DEFAULT_EXPORT = Path("outputs/12k_drive_end/figures")
FAULT_CHOICES = [ft.value for ft in FaultType]
LOAD_CHOICES = [lc.value for lc in LoadCondition]
DEFAULT_LOADS = [LoadCondition.HP_0.value, LoadCondition.HP_3.value]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualization assets for CWRU data")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing the dataset (default: 12k Drive End subset)",
    )
    parser.add_argument(
        "--fault-types",
        nargs="*",
        default=FAULT_CHOICES,
        choices=FAULT_CHOICES,
        help="Fault types to visualize (default: 全部)",
    )
    parser.add_argument(
        "--sensor",
        default=SensorLocation.DRIVE_END.value,
        choices=[sensor.value for sensor in SensorLocation],
        help="Sensor location to use for visualization",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4096,
        help="Number of samples used for short-time analyses",
    )
    parser.add_argument(
        "--loads",
        nargs="*",
        default=DEFAULT_LOADS,
        choices=LOAD_CHOICES,
        help="Load conditions to sample (default: 0hp + 3hp)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=DEFAULT_EXPORT,
        help="Directory to save generated figures",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    export_dir = args.export
    export_dir.mkdir(parents=True, exist_ok=True)

    sensor = SensorLocation(args.sensor)
    fault_types = [FaultType(ft) for ft in args.fault_types]
    load_conditions = [LoadCondition(load) for load in args.loads]
    stft_config = STFTConfig(nperseg=min(args.window, 2048))
    wavelet_config = WaveletConfig(max_scale=128)

    record_cache: Dict[Tuple[FaultType, LoadCondition], Segment] = {}

    for fault in fault_types:
        for load in load_conditions:
            record = _load_record(args.root, fault, sensor, load)
            if record is None:
                print(f"[WARN] 未找到 {fault.value}-{load.value} 的数据记录，跳过。")
                continue

            base_name = f"{fault.value}_{load.value}_{sensor.value}"
            segment = Segment(
                record=record,
                samples=record.values[: args.window],
                start_index=0,
            )
            record_cache[(fault, load)] = segment

            # Time-domain plot
            fig_time = plt.figure(figsize=(10, 6))
            ax_time = fig_time.add_subplot(111)
            plot_time_series(
                record,
                max_seconds=min(
                    record.values.size / record.sample_rate,
                    args.window / record.sample_rate,
                ),
                ax=ax_time,
            )
            ax_time.set_title(f"{fault.value.title()} ({load.value}) - 时间域波形")
            fig_time.savefig(export_dir / f"{base_name}_time.png", dpi=300)
            plt.close(fig_time)

            # Frequency domain plots
            fig_fft = plt.figure(figsize=(10, 6))
            ax_fft = fig_fft.add_subplot(111)
            plot_frequency_spectrum(
                segment,
                ax=ax_fft,
                title=f"{fault.value.title()} ({load.value}) - 幅度谱",
            )
            fig_fft.savefig(export_dir / f"{base_name}_spectrum.png", dpi=300)
            plt.close(fig_fft)

            fig_psd = plt.figure(figsize=(10, 6))
            ax_psd = fig_psd.add_subplot(111)
            plot_psd(segment, ax=ax_psd, title=f"{fault.value.title()} ({load.value}) - 功率谱密度")
            fig_psd.savefig(export_dir / f"{base_name}_psd.png", dpi=300)
            plt.close(fig_psd)

            # STFT
            fig_stft = plot_stft_figure(segment, config=stft_config)
            fig_stft.savefig(export_dir / f"{base_name}_stft.png", dpi=300)
            plt.close(fig_stft)

            # CWT
            fig_cwt = plot_cwt_figure(segment, config=wavelet_config)
            fig_cwt.savefig(export_dir / f"{base_name}_cwt.png", dpi=300)
            plt.close(fig_cwt)

            # EMD
            try:
                imfs = compute_emd(segment.samples)
            except RuntimeError as exc:
                print(f"[WARN] 无法计算 EMD ({exc})，跳过该图。")
            else:
                fig_emd = plot_emd_figure(imfs, sample_rate=record.sample_rate)
                fig_emd.savefig(export_dir / f"{base_name}_emd.png", dpi=300)
                plt.close(fig_emd)

            print(f"已生成 {fault.value}-{load.value} 的可视化图像。")

    # Multi-fault comparison figures per load
    for load in load_conditions:
        rows: List[Tuple[FaultType, Segment]] = []
        for fault in fault_types:
            key = (fault, load)
            segment = record_cache.get(key)
            if segment is not None:
                rows.append((fault, segment))

        if not rows:
            print(f"[WARN] {load.value} 负载下没有可用于对比的记录，跳过多子图输出。")
            continue

        fig, axes = plt.subplots(len(rows), 1, sharex=True, figsize=(10, 3 * len(rows)))
        if len(rows) == 1:
            axes = [axes]

        comparison_duration = min(
            args.window / rows[0][1].record.sample_rate,
            rows[0][1].record.values.size / rows[0][1].record.sample_rate,
        )

        for ax, (fault, segment) in zip(axes, rows):
            plot_time_series(
                segment.record,
                max_seconds=comparison_duration,
                ax=ax,
            )
            ax.set_title(f"{fault.value.title()} ({load.value}) - 时间域波形")

        fig.suptitle(f"{load.value} 负载下多故障时间域对比", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(export_dir / f"compare_{load.value}_{sensor.value}_time.png", dpi=300)
        plt.close(fig)
        print(f"已生成 {load.value} 负载下的多子图时间域对比图。")


def _load_record(root: Path, fault: FaultType, sensor: SensorLocation, load: LoadCondition):
    from cwru_analysis.data import load_catalog

    for record in load_catalog(root, fault_types=[fault], sensor=sensor, loads=[load]):
        return record
    return None


if __name__ == "__main__":
    main()
