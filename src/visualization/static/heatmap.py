"""Heatmap: benchmark x condition accuracy grid."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .style import (
    apply_style, save_figure, BENCHMARK_NAMES, CONDITION_NAMES, BENCHMARK_ORDER,
)


def plot_heatmap(df: pd.DataFrame, output_dir: str):
    """Create heatmap of accuracy across benchmarks and conditions.

    Args:
        df: Summary DataFrame with columns: benchmark, condition, accuracy.
        output_dir: Directory to save plots.
    """
    apply_style()

    benchmarks = [b for b in BENCHMARK_ORDER if b in df["benchmark"].unique()]
    conditions = sorted(df["condition"].unique(), key=lambda c: list(CONDITION_NAMES.keys()).index(c) if c in CONDITION_NAMES else 99)

    # Pivot to matrix
    pivot = df.pivot_table(values="accuracy", index="benchmark", columns="condition", aggfunc="first")
    pivot = pivot.reindex(index=benchmarks, columns=conditions) * 100

    # Rename for display
    pivot.index = [BENCHMARK_NAMES.get(b, b) for b in pivot.index]
    pivot.columns = [CONDITION_NAMES.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        vmin=0, vmax=100, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Accuracy (%)"},
    )
    ax.set_title("Cambrian-8B Accuracy: Benchmark x Condition")
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.tight_layout()
    save_figure(fig, output_dir, "heatmap")

    # Also create delta heatmap
    if "Normal" in pivot.columns:
        delta = pivot.subtract(pivot["Normal"], axis=0)
        delta = delta.drop(columns=["Normal"], errors="ignore")

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        sns.heatmap(
            delta, annot=True, fmt="+.1f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax2,
            cbar_kws={"label": "Accuracy Change (pp)"},
        )
        ax2.set_title("Accuracy Change from Normal Baseline")
        ax2.set_ylabel("")
        ax2.set_xlabel("")

        fig2.tight_layout()
        save_figure(fig2, output_dir, "heatmap_delta")
