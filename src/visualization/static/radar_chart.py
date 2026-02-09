"""Radar/spider chart: capability profile per condition across benchmarks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import (
    apply_style, save_figure, CONDITION_COLORS, BENCHMARK_NAMES,
    CONDITION_NAMES, BENCHMARK_ORDER,
)


def plot_radar(df: pd.DataFrame, output_dir: str):
    """Create radar chart showing model capability profile under each condition.

    Args:
        df: Summary DataFrame with columns: benchmark, condition, accuracy.
        output_dir: Directory to save plots.
    """
    apply_style()

    benchmarks = [b for b in BENCHMARK_ORDER if b in df["benchmark"].unique()]
    conditions = [c for c in CONDITION_COLORS if c in df["condition"].unique()]

    if len(benchmarks) < 3 or not conditions:
        return

    # Prepare angles
    n = len(benchmarks)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for condition in conditions:
        cond_data = df[df["condition"] == condition]
        values = []
        for bench in benchmarks:
            row = cond_data[cond_data["benchmark"] == bench]
            acc = row["accuracy"].values[0] * 100 if len(row) > 0 else 0
            values.append(acc)
        values += values[:1]  # Close

        color = CONDITION_COLORS.get(condition, "#999999")
        ax.plot(angles, values, "o-", linewidth=2, label=CONDITION_NAMES.get(condition, condition), color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([BENCHMARK_NAMES.get(b, b) for b in benchmarks])
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.set_title("Capability Profile Under Image Ablation", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, output_dir, "radar_chart")
