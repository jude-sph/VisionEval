"""Generate all visualizations from completed evaluation results.

Usage:
    python scripts/generate_plots.py
    python scripts/generate_plots.py --results_dir results --output_dir results/plots
"""

import os
import sys
import logging
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_plots")


def main(results_dir: str = "results", output_dir: str | None = None):
    """Generate all plots from evaluation results.

    Args:
        results_dir: Directory containing raw/ and aggregated/ results.
        output_dir: Output directory for plots. Defaults to results/plots/.
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "plots")

    from src.evaluation.results_store import ResultsStore
    from src.evaluation.metrics import compute_metrics

    store = ResultsStore(results_dir)

    # Aggregate results from all runs
    runs = store.list_completed_runs()
    if not runs:
        logger.error("No completed runs found. Run evaluations first.")
        return

    logger.info(f"Found {len(runs)} completed runs: {runs}")

    # Build summary
    summary = []
    for benchmark, condition in runs:
        results = store.load_results(benchmark, condition)
        if not results:
            continue

        # Determine scoring method
        from src.benchmarks import BENCHMARKS
        bench_cls = BENCHMARKS.get(benchmark)
        scoring = bench_cls.scoring_method if bench_cls else "mc_accuracy"

        metrics = compute_metrics(results, scoring)
        metrics["benchmark"] = benchmark
        metrics["condition"] = condition
        metrics["num_samples"] = len(results)
        summary.append(metrics)

    store.save_summary(summary)
    logger.info(f"Summary saved with {len(summary)} entries")

    # Generate static plots
    static_dir = os.path.join(output_dir, "static")
    os.makedirs(static_dir, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(summary)

    if not df.empty:
        from src.visualization.static.grouped_bar import plot_grouped_bar
        from src.visualization.static.delta_chart import plot_delta
        from src.visualization.static.heatmap import plot_heatmap
        from src.visualization.static.radar_chart import plot_radar
        from src.visualization.static.vulnerability import plot_vulnerability_index

        plot_grouped_bar(df, static_dir)
        plot_delta(df, static_dir)
        plot_heatmap(df, static_dir)
        plot_radar(df, static_dir)
        plot_vulnerability_index(df, static_dir)
        logger.info(f"Static plots saved to {static_dir}")

    # Generate interactive plots
    interactive_dir = os.path.join(output_dir, "interactive")
    os.makedirs(interactive_dir, exist_ok=True)

    if not df.empty:
        from src.visualization.interactive.dashboard import create_dashboard
        from src.visualization.interactive.sankey import create_sankey

        create_dashboard(df, interactive_dir)

        # Sankey needs per-question data
        all_results = {}
        for benchmark, condition in runs:
            key = (benchmark, condition)
            all_results[key] = store.load_results(benchmark, condition)
        create_sankey(all_results, interactive_dir)

        logger.info(f"Interactive plots saved to {interactive_dir}")

    logger.info("All plots generated!")


if __name__ == "__main__":
    fire.Fire(main)
