"""Generate global distribution visualizations (箱线图、直方图、雷达图等)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_features import _select_feature_columns, compute_fisher_scores
from cwru_analysis.data import Segment, generate_segments, load_catalog
from cwru_analysis.metadata import FaultType, LoadCondition, SensorLocation

FAULT_LABELS = {
    FaultType.NORMAL: "正常",
    FaultType.BALL: "滚动体故障",
    FaultType.INNER_RACE: "内圈故障",
    FaultType.OUTER_RACE: "外圈故障",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate distribution-level visualizations")
    parser.add_argument(
        "feature_csv",
        nargs="?",
        type=Path,
        default=Path("outputs/12k_drive_end/feature_table.csv"),
        help="Path to feature table",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("12k Drive End Bearing Fault Data"),
        help="Dataset root for sampling原始信号",
    )
    parser.add_argument(
        "--fault-types",
        nargs="*",
        default=[ft.value for ft in FaultType],
        choices=[ft.value for ft in FaultType],
        help="故障类别列表",
    )
    parser.add_argument(
        "--loads",
        nargs="*",
        default=[LoadCondition.HP_0.value, LoadCondition.HP_3.value],
        choices=[lc.value for lc in LoadCondition],
        help="覆盖的负载条件",
    )
    parser.add_argument(
        "--sensor",
        default=SensorLocation.DRIVE_END.value,
        choices=[sensor.value for sensor in SensorLocation],
        help="传感器位置",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2048,
        help="采样窗口长度",
    )
    parser.add_argument(
        "--segments-per-class",
        type=int,
        default=20,
        help="每个类别采集的窗口数量",
    )
    parser.add_argument(
        "--boxplot-topk",
        type=int,
        default=6,
        help="箱线图展示的特征数量 (按Fisher排序)",
    )
    parser.add_argument(
        "--radar-topk",
        type=int,
        default=6,
        help="雷达图展示的特征数量 (按Fisher排序)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("outputs/12k_drive_end/shape_figures"),
        help="图像输出目录",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.export.mkdir(parents=True, exist_ok=True)

    fault_types = [FaultType(ft) for ft in args.fault_types]
    load_conditions = [LoadCondition(load) for load in args.loads]
    sensor = SensorLocation(args.sensor)

    feature_df = pd.read_csv(args.feature_csv)
    feature_cols = _select_feature_columns(feature_df)
    fisher_df = compute_fisher_scores(feature_df, feature_cols, label_col="fault_type")

    box_features = fisher_df.head(args.boxplot_topk)["feature"].tolist()
    radar_features = fisher_df.head(args.radar_topk)["feature"].tolist()

    plot_boxplots(feature_df, box_features, args.export / "boxplot_features.png")
    plot_radar(feature_df, radar_features, args.export / "radar_features.png")

    samples = collect_signal_samples(
        root=args.root,
        fault_types=fault_types,
        loads=load_conditions,
        sensor=sensor,
        window=args.window,
        segments_per_class=args.segments_per_class,
    )
    if samples:
        plot_histograms(samples, args.export / "amplitude_histograms.png")
        plot_phase_space(samples, args.export / "phase_space.png")
    else:
        print("[WARN] 未采集到原始信号样本，跳过直方图/相空间绘制。")

    print(f"分布级图像已保存到 {args.export}")


def plot_boxplots(df: pd.DataFrame, features: List[str], out_path: Path) -> None:
    _apply_style()
    classes = df["fault_type"].unique().tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for idx, feature in enumerate(features):
        ax = axes.flat[idx]
        data = [df[df["fault_type"] == cls][feature] for cls in classes]
        labels = [translate_fault(cls) for cls in classes]
        ax.boxplot(data, labels=labels, notch=True, patch_artist=True)
        ax.set_title(_translate_feature(feature))
    for extra in axes.flat[len(features) :]:
        extra.axis("off")
    fig.suptitle("特征箱线图对比")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_radar(df: pd.DataFrame, features: List[str], out_path: Path) -> None:
    _apply_style()
    classes = df["fault_type"].unique().tolist()
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for cls in classes:
        values = df[df["fault_type"] == cls][features].mean()
        normed = (values - values.min()) / (values.max() - values.min() + 1e-12)
        data = np.concatenate([normed.values, normed.values[:1]])
        ax.plot(angles, data, label=translate_fault(cls))
        ax.fill(angles, data, alpha=0.2)
    ax.set_xticks(np.linspace(0, 2 * np.pi, len(features), endpoint=False))
    ax.set_xticklabels([_translate_feature(f) for f in features])
    ax.set_title("多域特征雷达图")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_histograms(samples: Dict[FaultType, np.ndarray], out_path: Path) -> None:
    _apply_style()
    plt.figure(figsize=(10, 6))
    bins = 60
    for fault, data in samples.items():
        plt.hist(data, bins=bins, alpha=0.4, density=True, label=translate_fault(fault))
    plt.title("幅值分布直方图")
    plt.xlabel("归一化幅值")
    plt.ylabel("概率密度")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_phase_space(samples: Dict[FaultType, np.ndarray], out_path: Path) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (fault, data) in zip(axes, samples.items()):
        if data.size < 3:
            continue
        x = data[:-1]
        y = data[1:]
        ax.scatter(x, y, s=5, alpha=0.4)
        ax.set_title(f"{translate_fault(fault)} 相空间")
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t+1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def collect_signal_samples(
    root: Path,
    fault_types: List[FaultType],
    loads: List[LoadCondition],
    sensor: SensorLocation,
    window: int,
    segments_per_class: int,
) -> Dict[FaultType, np.ndarray]:
    samples: Dict[FaultType, List[float]] = {}
    for fault in fault_types:
        collected: List[float] = []
        for load in loads:
            for record in load_catalog(root, fault_types=[fault], sensor=sensor, loads=[load]):
                for segment in generate_segments(record, window_size=window, step_size=window, normalize=True, centered=True):
                    collected.extend(segment.samples.tolist())
                    if len(collected) >= segments_per_class * window:
                        break
                if len(collected) >= segments_per_class * window:
                    break
        if collected:
            samples[fault] = np.array(collected[: segments_per_class * window])
    return samples


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei"],
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.unicode_minus": False,
        }
    )


def _translate_feature(feature: str) -> str:
    from analyze_features import _translate_feature_name

    return _translate_feature_name(feature)


def translate_fault(name: str | FaultType) -> str:
    fault = FaultType(name) if isinstance(name, str) else name
    return FAULT_LABELS.get(fault, fault.value)


if __name__ == "__main__":
    main()
