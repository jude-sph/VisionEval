"""Animated bar chart: watch accuracy change as conditions are applied."""

import os
import pandas as pd
import plotly.express as px

from src.visualization.static.style import (
    BENCHMARK_NAMES, CONDITION_NAMES, CONDITION_COLORS, BENCHMARK_ORDER,
)


def create_animated_bar(df: pd.DataFrame, output_dir: str):
    """Create animated bar chart transitioning between conditions.

    Args:
        df: Summary DataFrame with columns: benchmark, condition, accuracy.
        output_dir: Directory to save HTML.
    """
    benchmarks = [b for b in BENCHMARK_ORDER if b in df["benchmark"].unique()]
    conditions = [c for c in CONDITION_COLORS if c in df["condition"].unique()]

    if not benchmarks or not conditions:
        return

    # Build frame data
    rows = []
    for condition in conditions:
        cond_data = df[df["condition"] == condition]
        for bench in benchmarks:
            row = cond_data[cond_data["benchmark"] == bench]
            acc = row["accuracy"].values[0] * 100 if len(row) > 0 else 0
            rows.append({
                "Benchmark": BENCHMARK_NAMES.get(bench, bench),
                "Accuracy (%)": acc,
                "Condition": CONDITION_NAMES.get(condition, condition),
            })

    anim_df = pd.DataFrame(rows)

    fig = px.bar(
        anim_df,
        x="Benchmark",
        y="Accuracy (%)",
        color="Benchmark",
        animation_frame="Condition",
        range_y=[0, 100],
        title="Accuracy Under Different Conditions (Animated)",
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        showlegend=False,
    )

    # Slow down animation
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1500
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 800

    output_path = os.path.join(output_dir, "animated_bar.html")
    fig.write_html(output_path, include_plotlyjs=True)
