"""Per-question scatter: compare correctness across conditions."""

import os
import pandas as pd
import plotly.express as px

from src.visualization.static.style import BENCHMARK_NAMES, CONDITION_COLORS


def create_scatter_explorer(all_results: dict, output_dir: str):
    """Create interactive scatter plot comparing per-question outcomes.

    X-axis: correct under normal (1) vs wrong (0)
    Y-axis: correct under no_image (1) vs wrong (0)
    Jittered for visibility, colored by benchmark.

    Quadrants:
    - Top-right: Always correct (doesn't need image)
    - Bottom-right: Needs image (correct with, wrong without)
    - Top-left: Lucky without image (wrong with, correct without)
    - Bottom-left: Always wrong

    Args:
        all_results: Dict mapping (benchmark, condition) -> list of result dicts.
        output_dir: Directory to save HTML.
    """
    import numpy as np

    rows = []
    for bench in set(b for b, c in all_results.keys()):
        if (bench, "normal") not in all_results or (bench, "no_image") not in all_results:
            continue

        normal = {r["question_id"]: r["correct"] for r in all_results[(bench, "normal")]}
        no_img = {r["question_id"]: r["correct"] for r in all_results[(bench, "no_image")]}

        rng = np.random.RandomState(42)
        for qid in normal:
            if qid not in no_img:
                continue
            # Jitter for visibility
            jitter_x = rng.uniform(-0.15, 0.15)
            jitter_y = rng.uniform(-0.15, 0.15)

            n_correct = 1 if normal[qid] else 0
            ni_correct = 1 if no_img[qid] else 0

            quadrant = ""
            if n_correct and ni_correct:
                quadrant = "Always Correct"
            elif n_correct and not ni_correct:
                quadrant = "Needs Image"
            elif not n_correct and ni_correct:
                quadrant = "Lucky w/o Image"
            else:
                quadrant = "Always Wrong"

            rows.append({
                "benchmark": BENCHMARK_NAMES.get(bench, bench),
                "question_id": qid,
                "normal_correct": n_correct + jitter_x,
                "no_image_correct": ni_correct + jitter_y,
                "quadrant": quadrant,
            })

    if not rows:
        return

    scatter_df = pd.DataFrame(rows)

    fig = px.scatter(
        scatter_df,
        x="normal_correct",
        y="no_image_correct",
        color="benchmark",
        symbol="quadrant",
        hover_data=["question_id", "quadrant"],
        title="Per-Question Outcome: Normal vs No Image",
        labels={
            "normal_correct": "Correct with Image",
            "no_image_correct": "Correct without Image",
        },
        opacity=0.5,
    )

    # Add quadrant labels
    fig.add_annotation(x=1, y=1, text="Always Correct", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=1, y=0, text="Needs Image", showarrow=False, font=dict(size=14, color="red"))
    fig.add_annotation(x=0, y=1, text="Lucky w/o Image", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=0, y=0, text="Always Wrong", showarrow=False, font=dict(size=14, color="gray"))

    fig.update_layout(
        height=700, width=800,
        template="plotly_white",
        xaxis=dict(range=[-0.3, 1.3], tickvals=[0, 1], ticktext=["Wrong", "Correct"]),
        yaxis=dict(range=[-0.3, 1.3], tickvals=[0, 1], ticktext=["Wrong", "Correct"]),
    )

    # Add quadrant dividers
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)

    output_path = os.path.join(output_dir, "scatter_explorer.html")
    fig.write_html(output_path, include_plotlyjs=True)
