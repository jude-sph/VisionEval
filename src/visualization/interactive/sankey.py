"""Sankey diagram: flow from correct/incorrect under normal to other conditions."""

import os
import plotly.graph_objects as go

from src.visualization.static.style import BENCHMARK_NAMES, CONDITION_NAMES, CONDITION_COLORS


def create_sankey(all_results: dict, output_dir: str):
    """Create Sankey diagram showing how answers change between conditions.

    Shows flow: Normal Correct/Wrong -> No Image Correct/Wrong

    Args:
        all_results: Dict mapping (benchmark, condition) -> list of result dicts.
        output_dir: Directory to save HTML.
    """
    # Build per-question mapping for normal vs no_image
    benchmarks_with_both = set()
    for (bench, cond) in all_results:
        if cond == "normal":
            if (bench, "no_image") in all_results:
                benchmarks_with_both.add(bench)

    if not benchmarks_with_both:
        return

    # Aggregate across all benchmarks
    categories = {
        "correct_stays_correct": 0,
        "correct_becomes_wrong": 0,
        "wrong_stays_wrong": 0,
        "wrong_becomes_correct": 0,
    }

    for bench in sorted(benchmarks_with_both):
        normal_results = {r["question_id"]: r["correct"] for r in all_results[(bench, "normal")]}
        no_image_results = {r["question_id"]: r["correct"] for r in all_results[(bench, "no_image")]}

        for qid in normal_results:
            if qid not in no_image_results:
                continue
            normal_correct = normal_results[qid]
            noimg_correct = no_image_results[qid]

            if normal_correct and noimg_correct:
                categories["correct_stays_correct"] += 1
            elif normal_correct and not noimg_correct:
                categories["correct_becomes_wrong"] += 1
            elif not normal_correct and not noimg_correct:
                categories["wrong_stays_wrong"] += 1
            else:
                categories["wrong_becomes_correct"] += 1

    total = sum(categories.values())
    if total == 0:
        return

    # Sankey nodes
    labels = [
        f"Normal: Correct ({categories['correct_stays_correct'] + categories['correct_becomes_wrong']})",
        f"Normal: Wrong ({categories['wrong_stays_wrong'] + categories['wrong_becomes_correct']})",
        f"No Image: Correct ({categories['correct_stays_correct'] + categories['wrong_becomes_correct']})",
        f"No Image: Wrong ({categories['correct_becomes_wrong'] + categories['wrong_stays_wrong']})",
    ]

    # Links: source -> target -> value
    source = [0, 0, 1, 1]
    target = [2, 3, 2, 3]
    value = [
        categories["correct_stays_correct"],
        categories["correct_becomes_wrong"],
        categories["wrong_becomes_correct"],
        categories["wrong_stays_wrong"],
    ]
    link_colors = [
        "rgba(68, 119, 170, 0.4)",   # Correct -> Correct (blue)
        "rgba(238, 102, 119, 0.4)",   # Correct -> Wrong (red) - THESE NEEDED THE IMAGE
        "rgba(34, 136, 51, 0.4)",     # Wrong -> Correct (green) - lucky without image
        "rgba(170, 170, 170, 0.4)",   # Wrong -> Wrong (gray)
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=labels,
            color=["#4477AA", "#EE6677", "#4477AA", "#EE6677"],
        ),
        link=dict(source=source, target=target, value=value, color=link_colors),
    )])

    pct_broke = categories["correct_becomes_wrong"] / total * 100
    pct_lucky = categories["wrong_becomes_correct"] / total * 100

    fig.update_layout(
        title_text=(
            f"Answer Flow: Normal -> No Image "
            f"({pct_broke:.1f}% broke without image, {pct_lucky:.1f}% lucky without image)"
        ),
        height=500,
        template="plotly_white",
    )

    output_path = os.path.join(output_dir, "sankey.html")
    fig.write_html(output_path, include_plotlyjs=True)
