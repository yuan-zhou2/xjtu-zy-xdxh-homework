"""In-depth statistical analysis on generated feature tables."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced analytics on feature tables")
    parser.add_argument(
        "feature_csv",
        nargs="?",
        type=Path,
        default=Path("outputs/12k_drive_end/feature_table.csv"),
        help="Path to the feature table produced by run_full_analysis.py",
    )
    parser.add_argument(
        "--label-column",
        default="fault_type",
        help="Column name representing class labels",
    )
    parser.add_argument(
        "--load-column",
        default="load",
        help="Column name representing load conditions",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top discriminative features to report",
    )
    parser.add_argument(
        "--correlation-topk",
        type=int,
        default=15,
        help="Number of top features (by Fisher score) to include in the correlation heatmap",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to generate visualization figures based on the computed tables",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("outputs/12k_drive_end/analysis"),
        help="Directory to save analysis tables",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.feature_csv.exists():
        raise FileNotFoundError(args.feature_csv)
    export_dir = args.export
    export_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.feature_csv)
    label_col = args.label_column
    load_col = args.load_column
    feature_cols = _select_feature_columns(df)

    fisher_df = compute_fisher_scores(df, feature_cols, label_col)
    fisher_df.to_csv(export_dir / "fisher_scores.csv", index=False)

    cohens_df = compute_effect_sizes(df, feature_cols, label_col)
    cohens_df.to_csv(export_dir / "cohens_d.csv", index=False)

    load_df = compute_load_sensitivity(df, feature_cols, load_col)
    load_df.to_csv(export_dir / "load_sensitivity.csv", index=False)

    corr_df = compute_cross_domain_correlations(df, feature_cols)
    corr_df.to_csv(export_dir / "cross_domain_correlation.csv")

    if args.plot:
        figures_dir = export_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        _plot_topk_bar(
            fisher_df.head(args.topk),
            value_col="fisher_score",
            title="Fisher 得分 TOP",
            ylabel="Fisher 得分",
            out_path=figures_dir / "fisher_topk.png",
        )
        _plot_topk_bar(
            cohens_df.head(args.topk),
            value_col="abs_cohens_d",
            title="Cohen's d TOP",
            ylabel="|Cohen's d|",
            out_path=figures_dir / "cohens_d_topk.png",
        )
        _plot_topk_bar(
            load_df.head(args.topk),
            value_col="load_sensitivity",
            title="负载敏感性 TOP",
            ylabel="敏感性",
            out_path=figures_dir / "load_sensitivity_topk.png",
        )
        _plot_correlation_heatmap(
            corr_df,
            fisher_df.head(args.correlation_topk)["feature"].tolist(),
            out_path=figures_dir / "correlation_heatmap.png",
        )

    print("=== Top discriminative features (Fisher score) ===")
    print(fisher_df.head(args.topk))
    print("\n=== Top effect sizes (Cohen's d) ===")
    print(cohens_df.head(args.topk))
    print("\n=== Features most sensitive to load ===")
    print(load_df.head(args.topk))
    print(f"\nAnalysis tables saved to {export_dir}")
    if args.plot:
        print(f"Visualization figures saved to {figures_dir}")


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"segment_index", "rpm", "fault_type", "sensor", "load", "fault_size"}
    return [col for col in df.columns if col not in exclude]


def compute_fisher_scores(df: pd.DataFrame, features: List[str], label_col: str) -> pd.DataFrame:
    labels = df[label_col].unique()
    if labels.size < 2:
        raise ValueError("Need at least two classes for Fisher scores")
    global_mean = df[features].mean()
    scores = []
    for feature in features:
        numerator = 0.0
        denominator = 0.0
        for label in labels:
            subset = df[df[label_col] == label][feature]
            numerator += subset.size * (subset.mean() - global_mean[feature]) ** 2
            denominator += subset.var(ddof=1)
        score = numerator / (denominator + 1e-12)
        scores.append((feature, score))
    return pd.DataFrame(scores, columns=["feature", "fisher_score"]).sort_values(
        "fisher_score", ascending=False
    )


def compute_effect_sizes(df: pd.DataFrame, features: List[str], label_col: str) -> pd.DataFrame:
    labels = df[label_col].unique()
    if labels.size != 2:
        raise ValueError("Cohen's d computation currently supports exactly two classes")
    class_a, class_b = labels
    scores = []
    for feature in features:
        a = df[df[label_col] == class_a][feature]
        b = df[df[label_col] == class_b][feature]
        pooled_std = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
        d = (a.mean() - b.mean()) / (pooled_std + 1e-12)
        scores.append((feature, abs(d), d))
    result = pd.DataFrame(scores, columns=["feature", "abs_cohens_d", "signed_cohens_d"])
    return result.sort_values("abs_cohens_d", ascending=False)


def compute_load_sensitivity(df: pd.DataFrame, features: List[str], load_col: str) -> pd.DataFrame:
    loads = df[load_col].unique()
    scores = []
    for feature in features:
        total_var = df[feature].var(ddof=1)
        between = 0.0
        for load in loads:
            subset = df[df[load_col] == load][feature]
            between += subset.size * (subset.mean() - df[feature].mean()) ** 2
        between /= max(len(loads) - 1, 1)
        sensitivity = between / (total_var + 1e-12)
        scores.append((feature, sensitivity))
    return pd.DataFrame(scores, columns=["feature", "load_sensitivity"]).sort_values(
        "load_sensitivity", ascending=False
    )


def compute_cross_domain_correlations(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return df[features].corr(method="pearson")


def _apply_publication_style() -> None:
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


def _plot_topk_bar(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path) -> None:
    _apply_publication_style()
    plt.figure(figsize=(10, 6))
    labels = [_translate_feature_name(name) for name in df["feature"]]
    plt.barh(labels[::-1], df[value_col][::-1], color="#1f77b4")
    plt.title(title)
    plt.xlabel(ylabel)
    plt.ylabel("特征")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_correlation_heatmap(corr_df: pd.DataFrame, features: List[str], out_path: Path) -> None:
    subset = corr_df.loc[features, features]
    _apply_publication_style()
    plt.figure(figsize=(12, 10))
    im = plt.imshow(subset, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, label="Pearson 相关系数")
    translated = [_translate_feature_name(name) for name in features]
    plt.xticks(range(len(features)), translated, rotation=45, ha="right")
    plt.yticks(range(len(features)), translated)
    plt.title("特征相关性热力图")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _translate_feature_name(name: str) -> str:
    mapping = {
        "time_mean": "平均值",
        "time_std": "标准差",
        "time_variance": "方差",
        "time_rms": "均方根",
        "time_energy": "能量",
        "time_peak": "峰值",
        "time_peak_to_peak": "峰峰值",
        "time_skewness": "偏度",
        "time_kurtosis": "峭度",
        "time_crest_factor": "峰值因子",
        "time_clearance_factor": "裕度因子",
        "time_impulse_factor": "冲击因子",
        "time_shape_factor": "波形因子",
        "time_margin_factor": "余隙因子",
        "time_zero_crossings": "过零率",
        "time_hjorth_activity": "Hjorth 活动",
        "time_hjorth_mobility": "Hjorth 迁移",
        "time_hjorth_complexity": "Hjorth 复杂度",
        "freq_spectral_mean": "谱均值",
        "freq_spectral_var": "谱方差",
        "freq_spectral_std": "谱标准差",
        "freq_spectral_skewness": "谱偏度",
        "freq_spectral_kurtosis": "谱峭度",
        "freq_spectral_entropy": "谱熵",
        "freq_spectral_flatness": "谱平坦度",
        "freq_bandwidth": "带宽",
        "freq_peak_frequency": "峰值频率",
        "freq_roll_off_95": "滚降频率95%",
        "freq_envelope_peak_frequency": "包络峰频",
        "tf_stft_energy": "STFT 能量",
        "tf_stft_entropy": "STFT 熵",
        "tf_cwt_dominant_frequency": "CWT 主频",
        "tf_cwt_scale_entropy": "CWT 尺度熵",
        "tf_emd_imf_count": "EMD IMF 数",
        "tf_emd_energy_ratio": "EMD 能量占比",
        "tf_emd_hilbert_energy": "EMD Hilbert 能量",
    }
    return mapping.get(name, name)


if __name__ == "__main__":
    main()
