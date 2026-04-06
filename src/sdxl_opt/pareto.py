"""Pareto frontier analysis: find optimal quality–speed–memory trade-offs."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("sdxl_opt")

sns.set_theme(style="whitegrid", font_scale=1.1)


def find_pareto_frontier(
    df: pd.DataFrame,
    x_col: str = "latency_mean_s",
    y_col: str = "clip_score",
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> pd.DataFrame:
    """
    Find Pareto-optimal points. Default: minimize latency, maximize quality.
    Returns a subset of df that lies on the Pareto frontier.
    """
    points = df[[x_col, y_col]].values.copy()
    if not minimize_x:
        points[:, 0] = -points[:, 0]
    if not maximize_y:
        points[:, 1] = -points[:, 1]

    is_pareto = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not is_pareto[i]:
            continue
        for j, q in enumerate(points):
            if i == j or not is_pareto[j]:
                continue
            # q dominates p if q is better or equal on all axes and strictly better on at least one
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                is_pareto[i] = False
                break

    return df[is_pareto].copy()


def plot_pareto(
    df: pd.DataFrame,
    output_path: str = "results/pareto_frontier.png",
    x_col: str = "latency_mean_s",
    y_col: str = "clip_score",
    size_col: str = "peak_vram_gb",
) -> None:
    """
    Scatter plot: latency vs quality, point size = VRAM, Pareto frontier highlighted.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # All points
    sizes = df[size_col].values * 20  # scale for visibility
    scatter = ax.scatter(
        df[x_col], df[y_col],
        s=sizes, alpha=0.6, c="#6366f1", edgecolors="white", linewidth=1.5,
        zorder=3,
    )

    # Labels
    for _, row in df.iterrows():
        ax.annotate(
            row["config"],
            (row[x_col], row[y_col]),
            fontsize=8,
            ha="left",
            va="bottom",
            xytext=(6, 4),
            textcoords="offset points",
        )

    # Pareto frontier
    pareto = find_pareto_frontier(df, x_col, y_col)
    if len(pareto) > 1:
        pareto_sorted = pareto.sort_values(x_col)
        ax.plot(
            pareto_sorted[x_col], pareto_sorted[y_col],
            color="#ef4444", linewidth=2, linestyle="--", alpha=0.8,
            label="Pareto frontier", zorder=4,
        )
        ax.scatter(
            pareto_sorted[x_col], pareto_sorted[y_col],
            s=pareto_sorted[size_col].values * 20,
            c="#ef4444", edgecolors="white", linewidth=2,
            zorder=5,
        )

    ax.set_xlabel("Latency (s/image)", fontsize=12)
    ax.set_ylabel("CLIP Score", fontsize=12)
    ax.set_title("SDXL Compression: Quality vs Speed Trade-off", fontsize=14, fontweight="bold")

    # Size legend
    handles, labels = scatter.legend_elements(
        prop="sizes", num=3, func=lambda s: s / 20, fmt="{x:.1f} GB"
    )
    legend1 = ax.legend(handles, labels, loc="lower left", title="Peak VRAM")
    ax.add_artist(legend1)
    if len(pareto) > 1:
        ax.legend(loc="upper right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Pareto plot saved to {output_path}")


def plot_speedup_bar(
    df: pd.DataFrame,
    baseline_name: str = "baseline",
    output_path: str = "results/speedup_bar.png",
) -> None:
    """Bar chart of speedup relative to baseline."""
    baseline_latency = df.loc[df["config"] == baseline_name, "latency_mean_s"].values
    if len(baseline_latency) == 0:
        logger.warning("No baseline found, skipping speedup bar chart")
        return
    baseline_latency = baseline_latency[0]

    df = df.copy()
    df["speedup"] = baseline_latency / df["latency_mean_s"]
    df = df.sort_values("speedup", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.6)))

    colors = ["#ef4444" if s >= 3 else "#f59e0b" if s >= 1.5 else "#6366f1" for s in df["speedup"]]
    bars = ax.barh(df["config"], df["speedup"], color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, df["speedup"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}×", va="center", fontsize=10, fontweight="bold")

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Speedup vs Baseline", fontsize=12)
    ax.set_title("SDXL Optimization: Speedup Comparison", fontsize=14, fontweight="bold")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Speedup bar chart saved to {output_path}")


def plot_memory_comparison(
    df: pd.DataFrame,
    output_path: str = "results/memory_comparison.png",
) -> None:
    """Bar chart comparing peak VRAM usage."""
    df = df.sort_values("peak_vram_gb", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.6)))
    bars = ax.barh(df["config"], df["peak_vram_gb"], color="#10b981", edgecolor="white")

    for bar, val in zip(bars, df["peak_vram_gb"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} GB", va="center", fontsize=10)

    ax.set_xlabel("Peak VRAM (GB)", fontsize=12)
    ax.set_title("SDXL Optimization: Memory Usage", fontsize=14, fontweight="bold")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Memory comparison saved to {output_path}")
