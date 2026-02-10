"""Results storage: JSONL per-question predictions + CSV/JSON aggregated summaries."""

import json
import csv
from pathlib import Path
from typing import Optional
import pandas as pd


class ResultsStore:
    """Manages reading/writing evaluation results."""

    def __init__(self, results_dir: str = "results"):
        self.raw_dir = Path(results_dir) / "raw"
        self.agg_dir = Path(results_dir) / "aggregated"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.agg_dir.mkdir(parents=True, exist_ok=True)

    def _raw_path(self, benchmark: str, condition: str) -> Path:
        return self.raw_dir / f"{benchmark}_{condition}.jsonl"

    def get_completed_ids(self, benchmark: str, condition: str) -> set[str]:
        """Get question IDs already completed (for checkpoint/resume)."""
        path = self._raw_path(benchmark, condition)
        if not path.exists():
            return set()
        ids = set()
        with open(path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    ids.add(record["question_id"])
        return ids

    def append_result(self, benchmark: str, condition: str, result: dict) -> None:
        """Append a single prediction result to the JSONL file."""
        path = self._raw_path(benchmark, condition)
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")

    def load_results(self, benchmark: str, condition: str) -> list[dict]:
        """Load all predictions for a benchmark/condition pair."""
        path = self._raw_path(benchmark, condition)
        if not path.exists():
            return []
        results = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    def save_summary(self, summary: list[dict]) -> None:
        """Save aggregated summary as CSV and JSON."""
        # CSV
        csv_path = self.agg_dir / "summary.csv"
        if summary:
            keys = summary[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(summary)

        # JSON
        json_path = self.agg_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

    def get_summary_dataframe(self) -> Optional[pd.DataFrame]:
        """Load summary as pandas DataFrame."""
        csv_path = self.agg_dir / "summary.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None

    def list_completed_runs(self) -> list[tuple[str, str]]:
        """List all (benchmark, condition) pairs that have results."""
        # Known conditions (longest first so "gaussian_noise" matches before "noise")
        known_conditions = sorted(
            ["normal", "no_image", "wrong_image", "gaussian_noise",
             "heavy_blur", "shuffled_patches", "optimized_noise"],
            key=len, reverse=True,
        )
        runs = []
        for path in sorted(self.raw_dir.glob("*.jsonl")):
            stem = path.stem
            matched = False
            for cond in known_conditions:
                if stem.endswith(f"_{cond}"):
                    bench = stem[: -(len(cond) + 1)]
                    runs.append((bench, cond))
                    matched = True
                    break
            if not matched:
                # Fallback: split on last underscore
                parts = stem.rsplit("_", 1)
                if len(parts) == 2:
                    runs.append((parts[0], parts[1]))
        return runs
