"""Interactive Plotly dashboard: combined overview with filters."""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.visualization.static.style import (
    CONDITION_COLORS, BENCHMARK_NAMES, CONDITION_NAMES, BENCHMARK_ORDER, RANDOM_CHANCE,
)


def create_dashboard(df: pd.DataFrame, output_dir: str):
    """Create an interactive HTML dashboard combining multiple views.

    Args:
        df: Summary DataFrame with columns: benchmark, condition, accuracy.
        output_dir: Directory to save HTML file.
    """
    benchmarks = [b for b in BENCHMARK_ORDER if b in df["benchmark"].unique()]
    conditions = [c for c in CONDITION_COLORS if c in df["condition"].unique()]

    if not benchmarks or not conditions:
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Accuracy by Benchmark & Condition",
            "Accuracy Drop from Baseline",
            "Benchmark x Condition Heatmap",
            "Language Prior Vulnerability",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "bar"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # --- Plot 1: Grouped bar chart ---
    for condition in conditions:
        cond_data = df[df["condition"] == condition]
        accs = []
        bench_labels = []
        for bench in benchmarks:
            row = cond_data[cond_data["benchmark"] == bench]
            acc = row["accuracy"].values[0] * 100 if len(row) > 0 else 0
            accs.append(acc)
            bench_labels.append(BENCHMARK_NAMES.get(bench, bench))

        fig.add_trace(
            go.Bar(
                name=CONDITION_NAMES.get(condition, condition),
                x=bench_labels, y=accs,
                marker_color=CONDITION_COLORS.get(condition, "#999"),
                hovertemplate="%{x}: %{y:.1f}%<extra>" + CONDITION_NAMES.get(condition, condition) + "</extra>",
            ),
            row=1, col=1,
        )

    # --- Plot 2: Delta chart ---
    baselines = {}
    for bench in benchmarks:
        normal_row = df[(df["benchmark"] == bench) & (df["condition"] == "normal")]
        baselines[bench] = normal_row["accuracy"].values[0] * 100 if len(normal_row) > 0 else 0

    for condition in conditions:
        if condition == "normal":
            continue
        cond_data = df[df["condition"] == condition]
        deltas = []
        bench_labels = []
        for bench in benchmarks:
            row = cond_data[cond_data["benchmark"] == bench]
            acc = row["accuracy"].values[0] * 100 if len(row) > 0 else 0
            deltas.append(acc - baselines.get(bench, 0))
            bench_labels.append(BENCHMARK_NAMES.get(bench, bench))

        fig.add_trace(
            go.Bar(
                name=CONDITION_NAMES.get(condition, condition),
                x=bench_labels, y=deltas,
                marker_color=CONDITION_COLORS.get(condition, "#999"),
                showlegend=False,
                hovertemplate="%{x}: %{y:+.1f}pp<extra>" + CONDITION_NAMES.get(condition, condition) + "</extra>",
            ),
            row=1, col=2,
        )

    # --- Plot 3: Heatmap ---
    pivot = df.pivot_table(values="accuracy", index="benchmark", columns="condition", aggfunc="first")
    pivot = pivot.reindex(index=benchmarks, columns=conditions) * 100

    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=[CONDITION_NAMES.get(c, c) for c in pivot.columns],
            y=[BENCHMARK_NAMES.get(b, b) for b in pivot.index],
            colorscale="RdYlGn",
            zmin=0, zmax=100,
            text=pivot.values.round(1),
            texttemplate="%{text:.1f}%",
            hovertemplate="%{y} / %{x}: %{z:.1f}%<extra></extra>",
            showscale=True,
            colorbar=dict(title="Acc %", x=1.0),
        ),
        row=2, col=1,
    )

    # --- Plot 4: LPV Index ---
    if "normal" in df["condition"].values and "no_image" in df["condition"].values:
        lpv_data = []
        for bench in benchmarks:
            normal = df[(df["benchmark"] == bench) & (df["condition"] == "normal")]
            no_img = df[(df["benchmark"] == bench) & (df["condition"] == "no_image")]
            if len(normal) > 0 and len(no_img) > 0:
                normal_acc = normal["accuracy"].values[0]
                no_img_acc = no_img["accuracy"].values[0]
                lpv = 1 - (no_img_acc / normal_acc) if normal_acc > 0 else 0
                lpv_data.append((bench, lpv))

        lpv_data.sort(key=lambda x: x[1], reverse=True)

        fig.add_trace(
            go.Bar(
                x=[d[1] for d in lpv_data],
                y=[BENCHMARK_NAMES.get(d[0], d[0]) for d in lpv_data],
                orientation="h",
                marker_color=[f"rgb({int(255 * d[1])}, {int(255 * (1 - d[1]))}, 50)" for d in lpv_data],
                hovertemplate="%{y}: LPV=%{x:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=2, col=2,
        )

    # Layout
    fig.update_layout(
        title_text="VisionEval: Cambrian-8B Image Ablation Dashboard",
        height=900,
        barmode="group",
        template="plotly_white",
    )

    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Change (pp)", row=1, col=2)
    fig.update_xaxes(title_text="LPV Index", row=2, col=2)

    # Save
    output_path = os.path.join(output_dir, "dashboard.html")
    fig.write_html(output_path, include_plotlyjs=True)
