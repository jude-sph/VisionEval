"""Check evaluation progress across all benchmark/condition pairs.

Run this from any SSH session to see how far along the evaluation is.

Usage:
    python scripts/check_progress.py
    python scripts/check_progress.py --results_dir results
    python scripts/check_progress.py --watch          # Refresh every 30s
    python scripts/check_progress.py --watch --interval 10
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Expected sample counts per benchmark (from configs/benchmarks.yaml)
EXPECTED_SAMPLES = {
    "mmmu": 900,
    "mmbench": 3000,
    "scienceqa": 2000,
    "pope": 3000,
    "textvqa": 3000,
    "gqa": 5000,
}

# All planned runs
ALL_JOBS = [
    ("mmmu", "normal"), ("mmmu", "no_image"), ("mmmu", "wrong_image"), ("mmmu", "gaussian_noise"),
    ("mmbench", "normal"), ("mmbench", "no_image"), ("mmbench", "wrong_image"), ("mmbench", "gaussian_noise"),
    ("scienceqa", "normal"), ("scienceqa", "no_image"), ("scienceqa", "wrong_image"), ("scienceqa", "gaussian_noise"),
    ("pope", "normal"), ("pope", "no_image"), ("pope", "wrong_image"), ("pope", "gaussian_noise"),
    ("textvqa", "normal"), ("textvqa", "no_image"), ("textvqa", "wrong_image"), ("textvqa", "gaussian_noise"),
    ("gqa", "normal"), ("gqa", "no_image"), ("gqa", "wrong_image"), ("gqa", "gaussian_noise"),
]

CONDITION_NAMES = {
    "normal": "Normal",
    "no_image": "No Image",
    "wrong_image": "Wrong Img",
    "gaussian_noise": "Noise",
}

BENCHMARK_NAMES = {
    "mmmu": "MMMU",
    "mmbench": "MMBench",
    "scienceqa": "ScienceQA",
    "pope": "POPE",
    "textvqa": "TextVQA",
    "gqa": "GQA",
}


def count_lines(path: Path) -> int:
    """Count lines in a file without loading it all into memory."""
    if not path.exists():
        return 0
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def get_run_stats(jsonl_path: Path) -> dict:
    """Get statistics from a JSONL results file."""
    if not jsonl_path.exists():
        return {"done": 0, "correct": 0, "avg_ms": 0, "last_update": None}

    done = 0
    correct = 0
    total_ms = 0
    last_update = None

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                done += 1
                if record.get("correct"):
                    correct += 1
                total_ms += record.get("inference_time_ms", 0)
            except json.JSONDecodeError:
                continue

    # Get file modification time as last update
    if done > 0:
        last_update = datetime.fromtimestamp(jsonl_path.stat().st_mtime)

    return {
        "done": done,
        "correct": correct,
        "avg_ms": total_ms / done if done > 0 else 0,
        "last_update": last_update,
    }


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds <= 0:
        return "-"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def format_time_ago(dt: datetime | None) -> str:
    """Format a datetime as 'X ago'."""
    if dt is None:
        return "-"
    delta = datetime.now() - dt
    seconds = delta.total_seconds()
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m ago"


def is_running(logs_dir: Path) -> bool:
    """Check if an evaluation process appears to be running."""
    pid_file = logs_dir / "eval.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if process exists (doesn't actually kill)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            pass

    # Also check tmux
    try:
        import subprocess
        result = subprocess.run(
            ["tmux", "has-session", "-t", "visioneval"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def print_progress(results_dir: str = "results"):
    """Print a progress table for all evaluation runs."""
    raw_dir = Path(results_dir) / "raw"
    logs_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "logs"

    # Check if process is running
    running = is_running(logs_dir)
    status_icon = "RUNNING" if running else "STOPPED"
    status_color = "\033[92m" if running else "\033[91m"
    reset = "\033[0m"

    print()
    print(f"  VisionEval Progress  [{status_color}{status_icon}{reset}]")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Header
    header = f"  {'Benchmark':<12} {'Condition':<12} {'Progress':>14} {'%':>6} {'Acc':>7} {'Avg/q':>8} {'ETA':>8} {'Updated':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_done = 0
    total_expected = 0
    total_correct = 0
    active_runs = []

    for benchmark, condition in ALL_JOBS:
        jsonl_path = raw_dir / f"{benchmark}_{condition}.jsonl"
        expected = EXPECTED_SAMPLES.get(benchmark, 0)
        total_expected += expected

        stats = get_run_stats(jsonl_path)
        done = stats["done"]
        total_done += done
        total_correct += stats["correct"]

        # Calculate progress
        pct = (done / expected * 100) if expected > 0 else 0
        acc = (stats["correct"] / done * 100) if done > 0 else 0
        avg_s = stats["avg_ms"] / 1000 if stats["avg_ms"] > 0 else 0
        remaining = expected - done
        eta_s = remaining * avg_s if avg_s > 0 else 0

        # Status indicator
        if done >= expected and expected > 0:
            status = "\033[92m DONE \033[0m"  # Green
        elif done > 0:
            status = "\033[93m >>>  \033[0m"  # Yellow (in progress)
            active_runs.append((benchmark, condition, pct, eta_s))
        else:
            status = "      "  # Not started

        bench_display = BENCHMARK_NAMES.get(benchmark, benchmark)
        cond_display = CONDITION_NAMES.get(condition, condition)
        progress_str = f"{done:>5}/{expected:<5}"
        avg_str = f"{avg_s:.1f}s" if avg_s > 0 else "-"
        eta_str = format_duration(eta_s) if done > 0 and done < expected else ("-" if done == 0 else "done")
        updated_str = format_time_ago(stats["last_update"])

        print(
            f"{status} {bench_display:<12} {cond_display:<12} "
            f"{progress_str:>14} {pct:>5.1f}% {acc:>6.1f}% "
            f"{avg_str:>8} {eta_str:>8} {updated_str:>12}"
        )

    print("  " + "-" * (len(header) - 2))

    # Summary
    overall_pct = (total_done / total_expected * 100) if total_expected > 0 else 0
    overall_acc = (total_correct / total_done * 100) if total_done > 0 else 0
    print(f"  {'TOTAL':<12} {'':12} {total_done:>5}/{total_expected:<5} {overall_pct:>5.1f}% {overall_acc:>6.1f}%")

    # ETA for active runs
    if active_runs:
        max_eta = max(r[3] for r in active_runs)
        print(f"\n  Longest active ETA: {format_duration(max_eta)}")

    # Log file hint
    log_file = logs_dir / "eval.log"
    if log_file.exists():
        print(f"\n  Log: tail -f {log_file}")

    print()


def main(results_dir: str = "results", watch: bool = False, interval: int = 30):
    """Show evaluation progress.

    Args:
        results_dir: Directory containing results.
        watch: If True, refresh automatically.
        interval: Refresh interval in seconds (with --watch).
    """
    if watch:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print_progress(results_dir)
                print(f"  Refreshing every {interval}s. Press Ctrl+C to stop.")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        print_progress(results_dir)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
