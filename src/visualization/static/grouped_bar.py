"""Grouped bar chart: accuracy per benchmark, grouped by condition."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import (
    apply_style, save_figure, CONDITION_COLORS, BENCHMARK_NAMES,
    CONDITION_NAMES, RANDOM_CHANCE, BENCHMARK_ORDER,
)


def plot_grouped_bar(df: pd.DataFrame, output_dir: str):
    """Create grouped bar chart of accuracy across benchmarks and conditions.

    Args:
        df: Summary DataFrame with columns: benchmark, condition, accuracy.
        output_dir: Directory to save plots.
    """
    apply_style()

    # Filter to available benchmarks and order them
    benchmarks = [b for b in BENCHMARK_ORDER if b in df["benchmark"].unique()]
    conditions = [c for c in CONDITION_COLORS if c in df["condition"].unique()]

    if not benchmarks or not conditions:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    n_benchmarks = len(benchmarks)
    n_conditions = len(conditions)
    bar_width = 0.8 / n_conditions
    x = np.arange(n_benchmarks)

    for i, condition in enumerate(conditions):
        cond_data = df[df["condition"] == condition]
        accuracies = []
        for bench in benchmarks:
            row = cond_data[cond_data["benchmark"] == bench]
            acc = row["accuracy"].values[0] * 100 if len(row) > 0 else 0
            accuracies.append(acc)

        offset = (i - n_conditions / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, accuracies, bar_width,
            label=CONDITION_NAMES.get(condition, condition),
            color=CONDITION_COLORS.get(condition, "#999999"),
            edgecolor="white", linewidth=0.5,
        )

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{acc:.1f}", ha="center", va="bottom", fontsize=7,
                )

    # Random chance lines
    for i, bench in enumerate(benchmarks):
        chance = RANDOM_CHANCE.get(bench, 0) * 100
        if chance > 0:
            ax.hlines(
                chance, i - 0.45, i + 0.45,
                colors="gray", linestyles="dashed", linewidth=0.8, alpha=0.6,
            )

    # Add vertical separator between high/low prior groups
    separator_idx = None
    for i, bench in enumerate(benchmarks):
        if i > 0:
            from .style import HIGH_PRIOR, LOW_PRIOR
            prev_is_high = benchmarks[i - 1] in HIGH_PRIOR
            curr_is_low = bench in LOW_PRIOR
            if prev_is_high and curr_is_low:
                separator_idx = i - 0.5

    if separator_idx is not None:
        ax.axvline(separator_idx, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(separator_idx - 0.3, ax.get_ylim()[1] * 0.95, "High Prior", ha="right", fontsize=8, color="gray")
        ax.text(separator_idx + 0.3, ax.get_ylim()[1] * 0.95, "Low Prior", ha="left", fontsize=8, color="gray")

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cambrian-8B Accuracy Under Image Ablation Conditions")
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_NAMES.get(b, b) for b in benchmarks])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, output_dir, "grouped_bar")
