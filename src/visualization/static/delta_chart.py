"""Delta chart: accuracy drop from baseline (normal) per condition."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import (
    apply_style, save_figure, CONDITION_COLORS, BENCHMARK_NAMES,
    CONDITION_NAMES, BENCHMARK_ORDER,
)


def plot_delta(df: pd.DataFrame, output_dir: str):
    """Create chart showing accuracy drop from normal baseline.

    Args:
        df: Summary DataFrame with columns: benchmark, condition, accuracy.
        output_dir: Directory to save plots.
    """
    apply_style()

    benchmarks = [b for b in BENCHMARK_ORDER if b in df["benchmark"].unique()]
    conditions = [c for c in CONDITION_COLORS if c in df["condition"].unique() and c != "normal"]

    if not benchmarks or not conditions:
        return

    # Get baseline accuracies
    baselines = {}
    for bench in benchmarks:
        normal_row = df[(df["benchmark"] == bench) & (df["condition"] == "normal")]
        baselines[bench] = normal_row["accuracy"].values[0] * 100 if len(normal_row) > 0 else 0

    fig, ax = plt.subplots(figsize=(12, 6))

    n_benchmarks = len(benchmarks)
    n_conditions = len(conditions)
    bar_width = 0.8 / n_conditions
    x = np.arange(n_benchmarks)

    for i, condition in enumerate(conditions):
        cond_data = df[df["condition"] == condition]
        deltas = []
        for bench in benchmarks:
            row = cond_data[cond_data["benchmark"] == bench]
            acc = row["accuracy"].values[0] * 100 if len(row) > 0 else 0
            delta = acc - baselines.get(bench, 0)
            deltas.append(delta)

        offset = (i - n_conditions / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, deltas, bar_width,
            label=CONDITION_NAMES.get(condition, condition),
            color=CONDITION_COLORS.get(condition, "#999999"),
            edgecolor="white", linewidth=0.5,
        )

        for bar, delta in zip(bars, deltas):
            y_pos = bar.get_height() - 1.5 if delta < 0 else bar.get_height() + 0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos,
                f"{delta:+.1f}", ha="center", va="top" if delta < 0 else "bottom",
                fontsize=7,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Accuracy Change (pp)")
    ax.set_title("Accuracy Drop from Normal Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_NAMES.get(b, b) for b in benchmarks])
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, output_dir, "delta_chart")
