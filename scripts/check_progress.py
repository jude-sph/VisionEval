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
    "mmbench": 4329,  # lmms-lab/MMBench en/dev split
    "pope": 9000,
    "textvqa": 3000,
    # Temporarily disabled — add back if needed:
    # "gqa": 5000,
    # "scienceqa": 2000,
}

# All planned runs
ALL_JOBS = [
    ("mmmu", "normal"), ("mmmu", "no_image"), ("mmmu", "wrong_image"), ("mmmu", "gaussian_noise"),
    ("mmbench", "normal"), ("mmbench", "no_image"), ("mmbench", "wrong_image"), ("mmbench", "gaussian_noise"),
    ("pope", "normal"), ("pope", "no_image"), ("pope", "wrong_image"), ("pope", "gaussian_noise"),
    ("textvqa", "normal"), ("textvqa", "no_image"), ("textvqa", "wrong_image"), ("textvqa", "gaussian_noise"),
    # Temporarily disabled — add back if needed:
    # ("gqa", "normal"), ("gqa", "no_image"), ("gqa", "wrong_image"), ("gqa", "gaussian_noise"),
    # ("scienceqa", "normal"), ("scienceqa", "no_image"), ("scienceqa", "wrong_image"), ("scienceqa", "gaussian_noise"),
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
        return {"done": 0, "correct": 0, "errors": 0, "valid": 0, "avg_ms": 0, "last_update": None}

    done = 0
    correct = 0
    errors = 0
    total_ms = 0
    last_update = None

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                done += 1
                if record.get("error"):
                    errors += 1
                elif record.get("correct"):
                    correct += 1
                total_ms += record.get("inference_time_ms", 0)
            except json.JSONDecodeError:
                continue

    # Get file modification time as last update
    if done > 0:
        last_update = datetime.fromtimestamp(jsonl_path.stat().st_mtime)

    # Accuracy excludes error samples
    valid = done - errors

    return {
        "done": done,
        "correct": correct,
        "errors": errors,
        "valid": valid,
        "avg_ms": total_ms / valid if valid > 0 else 0,
        "last_update": last_update,
    }


def format_duration(seconds: float, precise: bool = False) -> str:
    """Format seconds into human-readable duration.

    Args:
        seconds: Duration in seconds.
        precise: If True, always include seconds component.
    """
    if seconds <= 0:
        return "-"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        if precise:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


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

    # --- Read all stats once (avoid re-reading files 3+ times per job) ---
    all_stats = {}  # (benchmark, condition) -> stats dict
    for benchmark, condition in ALL_JOBS:
        jsonl_path = raw_dir / f"{benchmark}_{condition}.jsonl"
        all_stats[(benchmark, condition)] = get_run_stats(jsonl_path)

    # --- Compute per-benchmark avg time (weighted across all conditions with data) ---
    bench_avg_s = {}  # benchmark -> avg seconds per question
    for benchmark in EXPECTED_SAMPLES:
        total_valid = 0
        total_time_ms = 0
        for b, c in ALL_JOBS:
            if b == benchmark:
                s = all_stats[(b, c)]
                if s["valid"] > 0 and s["avg_ms"] > 0:
                    total_valid += s["valid"]
                    total_time_ms += s["avg_ms"] * s["valid"]
        if total_valid > 0:
            bench_avg_s[benchmark] = (total_time_ms / total_valid) / 1000

    # Global average fallback (for benchmarks with no data at all)
    all_avg_s = sum(bench_avg_s.values()) / len(bench_avg_s) if bench_avg_s else 0

    # Header
    header = f"  {'Benchmark':<12} {'Condition':<12} {'Progress':>14} {'%':>6} {'Acc':>7} {'Avg/q':>8} {'ETA':>10} {'Updated':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_done = 0
    total_expected = 0
    total_correct = 0
    total_valid = 0
    active_run = None  # Track the currently active (in-progress) job
    earliest_update = None  # Track when the run started (first file modified)

    for benchmark, condition in ALL_JOBS:
        expected = EXPECTED_SAMPLES.get(benchmark, 0)
        total_expected += expected

        stats = all_stats[(benchmark, condition)]
        done = stats["done"]
        valid = stats["valid"]
        total_done += done
        total_correct += stats["correct"]
        total_valid += valid

        # Track earliest file modification for elapsed time
        if stats["last_update"] is not None:
            if earliest_update is None:
                earliest_update = stats["last_update"]
            else:
                earliest_update = min(earliest_update, stats["last_update"])

        # Calculate progress
        pct = (done / expected * 100) if expected > 0 else 0
        acc = (stats["correct"] / valid * 100) if valid > 0 else 0
        avg_s = stats["avg_ms"] / 1000 if stats["avg_ms"] > 0 else 0
        remaining = expected - done
        eta_s = remaining * avg_s if avg_s > 0 else 0

        # Status indicator
        if done >= expected and expected > 0:
            status = "\033[92m DONE \033[0m"  # Green
        elif done > 0 and done < expected:
            status = "\033[93m >>>  \033[0m"  # Yellow (in progress)
            active_run = (benchmark, condition, done, expected, avg_s, eta_s)
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
            f"{avg_str:>8} {eta_str:>10} {updated_str:>12}"
        )

    print("  " + "-" * (len(header) - 2))

    # Summary
    overall_pct = (total_done / total_expected * 100) if total_expected > 0 else 0
    overall_acc = (total_correct / total_valid * 100) if total_valid > 0 else 0
    print(f"  {'TOTAL':<12} {'':12} {total_done:>5}/{total_expected:<5} {overall_pct:>5.1f}% {overall_acc:>6.1f}%")

    # --- Per-benchmark ETAs (using cached stats) ---
    # Compute per-benchmark and total remaining time
    total_remaining_s = 0
    bench_remaining = {}  # benchmark -> remaining seconds
    bench_done_counts = {}  # benchmark -> (done, total_expected, conditions_done, conditions_total)

    for benchmark in EXPECTED_SAMPLES:
        expected = EXPECTED_SAMPLES[benchmark]
        conds_done = 0
        conds_total = 0
        b_done = 0

        for b, c in ALL_JOBS:
            if b == benchmark:
                conds_total += 1
                s = all_stats[(b, c)]
                b_done += s["done"]
                if s["done"] >= expected:
                    conds_done += 1

                remaining_q = max(0, expected - s["done"])
                if remaining_q == 0:
                    job_eta = 0
                else:
                    avg = bench_avg_s.get(benchmark, all_avg_s)
                    job_eta = remaining_q * avg

                total_remaining_s += job_eta
                bench_remaining[benchmark] = bench_remaining.get(benchmark, 0) + job_eta

        bench_done_counts[benchmark] = (b_done, expected * conds_total, conds_done, conds_total)

    # Print per-benchmark ETAs
    print()
    print(f"  {'Benchmark ETAs':}")
    for benchmark in EXPECTED_SAMPLES:
        bench_display = BENCHMARK_NAMES.get(benchmark, benchmark)
        b_done, b_expected, conds_done, conds_total = bench_done_counts[benchmark]
        bench_pct = b_done / b_expected * 100 if b_expected > 0 else 0
        remaining_s = bench_remaining.get(benchmark, 0)

        # Progress bar (20 chars wide)
        bar_width = 20
        filled = int(bar_width * bench_pct / 100)
        bar = "=" * filled + (">" if filled < bar_width else "") + " " * max(0, bar_width - filled - 1)

        avg_str = ""
        if benchmark in bench_avg_s:
            avg_str = f"  ({bench_avg_s[benchmark]:.1f}s/q)"

        if remaining_s > 0:
            eta_str = format_duration(remaining_s)
            print(f"  {bench_display:<12} [{bar}] {conds_done}/{conds_total} conditions   ~{eta_str} remaining{avg_str}")
        else:
            if conds_done == conds_total:
                print(f"  {bench_display:<12} [{'=' * bar_width}] {conds_done}/{conds_total} conditions   done")
            else:
                print(f"  {bench_display:<12} [{bar}] {conds_done}/{conds_total} conditions   estimating...")

    # Active run callout
    if active_run and running:
        a_bench, a_cond, a_done, a_expected, a_avg, a_eta = active_run
        a_bench_display = BENCHMARK_NAMES.get(a_bench, a_bench)
        a_cond_display = CONDITION_NAMES.get(a_cond, a_cond)
        print(f"\n  Currently running:  {a_bench_display}/{a_cond_display}  ({a_done}/{a_expected}, ~{format_duration(a_eta)} left)")

    # Elapsed time
    if earliest_update and total_done > 0:
        # Approximate start time from earliest file mod and avg speed
        # Better: just show wall time since first result was written
        elapsed = (datetime.now() - earliest_update).total_seconds()
        if elapsed > 0:
            rate = total_done / elapsed
            print(f"  Elapsed since first result:  {format_duration(elapsed)}  ({rate:.1f} q/s)")

    # Total system ETA
    if total_remaining_s > 0:
        completion_time = datetime.now() + timedelta(seconds=total_remaining_s)
        print(f"\n  Total estimated remaining:  {format_duration(total_remaining_s, precise=True)}")
        print(f"  Estimated completion:       {completion_time.strftime('%Y-%m-%d %H:%M')}")
    elif total_done > 0 and total_done >= total_expected:
        print(f"\n  All benchmark evaluations complete!")

    # Noise optimization section
    opt_dir = Path(results_dir) / "optimization"
    pixel_dir = opt_dir / "pixel"
    opt_files = sorted(opt_dir.glob("*_optimized_embeddings.jsonl")) if opt_dir.exists() else []

    # Also check if noise optimization is running (tmux session or PID)
    noise_running = False
    noise_pid_file = logs_dir / "noise.pid"
    if noise_pid_file.exists():
        try:
            pid = int(noise_pid_file.read_text().strip())
            os.kill(pid, 0)
            noise_running = True
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    if not noise_running:
        try:
            import subprocess
            result = subprocess.run(
                ["tmux", "has-session", "-t", "noise"],
                capture_output=True, timeout=5,
            )
            noise_running = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Find embedding result files
    universal_files = sorted(opt_dir.glob("*_universal_embeddings.jsonl")) if opt_dir.exists() else []
    universal_summaries = sorted(opt_dir.glob("*_universal_summary.json")) if opt_dir.exists() else []

    # Find pixel result files
    pixel_per_q_files = sorted(pixel_dir.glob("*_pixel_optimized.jsonl")) if pixel_dir.exists() else []
    pixel_universal_summaries = sorted(pixel_dir.glob("*_pixel_universal_summary.json")) if pixel_dir.exists() else []

    has_any_noise = opt_files or universal_files or universal_summaries or pixel_per_q_files or pixel_universal_summaries or noise_running

    if has_any_noise:
        noise_status = "\033[92mRUNNING\033[0m" if noise_running else "\033[91mSTOPPED\033[0m"
        print()
        print(f"  Noise Optimization  [{noise_status}]")

        # --- Embedding Universal results ---
        for summary_path in universal_summaries:
            bench_name = summary_path.stem.replace("_universal_summary", "")
            bench_display = BENCHMARK_NAMES.get(bench_name, bench_name)
            try:
                with open(summary_path) as f:
                    s = json.loads(f.read())
                acc = s.get("accuracy", 0)
                epochs = s.get("num_epochs", 0)
                n = s.get("num_samples", 0)
                init_loss = s.get("initial_avg_loss", 0)
                final_loss = s.get("final_avg_loss", 0)
                opt_time = s.get("optimization_time_s", 0)
                print(f"  Emb Universal  {bench_display:<10} {n} samples, {epochs} epochs: "
                      f"acc={acc:.1f}%  loss {init_loss:.3f}->{final_loss:.3f}  ({opt_time:.0f}s)")
            except (json.JSONDecodeError, KeyError):
                pass

        # --- Pixel Universal results ---
        for summary_path in pixel_universal_summaries:
            bench_name = summary_path.stem.replace("_pixel_universal_summary", "")
            bench_display = BENCHMARK_NAMES.get(bench_name, bench_name)
            try:
                with open(summary_path) as f:
                    s = json.loads(f.read())
                acc = s.get("accuracy", 0)
                epochs = s.get("num_epochs", 0)
                n = s.get("num_samples", 0)
                init_loss = s.get("initial_avg_loss", 0)
                final_loss = s.get("final_avg_loss", 0)
                opt_time = s.get("optimization_time_s", 0)
                print(f"  Pix Universal  {bench_display:<10} {n} samples, {epochs} epochs: "
                      f"acc={acc:.1f}%  loss {init_loss:.3f}->{final_loss:.3f}  ({opt_time:.0f}s)")
            except (json.JSONDecodeError, KeyError):
                pass

        # Check if universal is in progress (summary doesn't exist yet but log mentions it)
        if noise_running and not universal_summaries and not pixel_universal_summaries:
            noise_log = logs_dir / "optimize_noise.log"
            if noise_log.exists():
                try:
                    with open(noise_log, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 4096))
                        tail = f.read().decode("utf-8", errors="ignore")
                    for line in reversed(tail.splitlines()):
                        if "Epoch " in line and "avg_loss=" in line:
                            epoch_part = line.split("Epoch ")[-1].split(":")[0]
                            loss_part = line.split("avg_loss=")[-1].split(" ")[0]
                            mode_prefix = "Pixel" if "PIXEL" in tail else "Emb"
                            print(f"  {mode_prefix} Universal  (in progress)  Epoch {epoch_part}  avg_loss={loss_part}")
                            break
                except OSError:
                    pass

        # --- Per-question results table (embedding + pixel) ---
        all_per_q = []
        for opt_path in opt_files:
            all_per_q.append(("Emb per-q", opt_path, "_optimized_embeddings", "_optimized_summary"))
        for opt_path in pixel_per_q_files:
            all_per_q.append(("Pix per-q", opt_path, "_pixel_optimized", "_pixel_optimized_summary"))

        if all_per_q:
            opt_header = f"  {'Mode':<14} {'Benchmark':<10} {'Progress':>10} {'Acc':>7} {'Avg Loss':>10} {'Loss Drop':>10} {'NaN':>5} {'Avg/q':>8} {'ETA':>10} {'Updated':>12}"
            print(opt_header)
            print("  " + "-" * (len(opt_header) - 2))

        for mode_label, opt_path, stem_suffix, summary_suffix in all_per_q:
            bench_name = opt_path.stem.replace(stem_suffix, "")
            bench_display = BENCHMARK_NAMES.get(bench_name, bench_name)

            done = 0
            correct = 0
            nan_count = 0
            total_loss_init = 0
            total_loss_final = 0
            valid_loss_count = 0
            total_time = 0
            expected = 50

            with open(opt_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                        done += 1
                        if r.get("correct"):
                            correct += 1
                        if r.get("nan_detected"):
                            nan_count += 1
                        total_loss_init += r.get("initial_loss", 0)
                        if r.get("final_loss") is not None:
                            total_loss_final += r["final_loss"]
                            valid_loss_count += 1
                        total_time += r.get("optimization_time_s", 0)
                    except json.JSONDecodeError:
                        continue

            # Check summary file for expected count
            search_dir = pixel_dir if "Pix" in mode_label else opt_dir
            summary_path = search_dir / f"{bench_name}{summary_suffix}.json"
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary = json.loads(f.read())
                    expected = summary.get("num_samples", expected)
                except (json.JSONDecodeError, KeyError):
                    pass

            last_update = datetime.fromtimestamp(opt_path.stat().st_mtime) if done > 0 else None

            acc = (correct / done * 100) if done > 0 else 0
            avg_loss = (total_loss_final / valid_loss_count) if valid_loss_count > 0 else 0
            avg_drop = ((total_loss_init - total_loss_final) / valid_loss_count) if valid_loss_count > 0 else 0
            avg_time = (total_time / done) if done > 0 else 0
            opt_remaining = (expected - done) * avg_time if done > 0 and done < expected else 0
            opt_eta_str = format_duration(opt_remaining) if done > 0 and done < expected else ("-" if done == 0 else "done")

            if done >= expected and expected > 0:
                status = "\033[92m DONE \033[0m"
            elif done > 0:
                status = "\033[93m >>>  \033[0m"
            else:
                status = "      "

            nan_str = str(nan_count) if nan_count > 0 else "-"

            print(
                f"{status} {mode_label:<14} {bench_display:<10} "
                f"{done:>4}/{expected:<4} "
                f"{acc:>6.1f}% "
                f"{avg_loss:>9.3f} "
                f"{avg_drop:>+9.3f} "
                f"{nan_str:>5} "
                f"{avg_time:>7.1f}s "
                f"{opt_eta_str:>10} "
                f"{format_time_ago(last_update):>12}"
            )

        if all_per_q:
            print("  " + "-" * (len(opt_header) - 2))

        # Count pixel images saved
        if pixel_dir.exists():
            pixel_images = list((pixel_dir / "images").glob("*.png")) if (pixel_dir / "images").exists() else []
            universal_pngs = list(pixel_dir.glob("*_pixel_universal*.png"))
            if pixel_images or universal_pngs:
                print(f"  Pixel images: {len(pixel_images)} per-question + {len(universal_pngs)} universal")

        # Noise optimization log hint
        noise_log = logs_dir / "optimize_noise.log"
        if noise_log.exists():
            print(f"  Log: tail -f {noise_log}")

    # --- Bottleneck Optimization section ---
    two_bit_dir = Path(results_dir) / "two_bit"

    # Check if bottleneck is running (tmux or PID)
    bottleneck_running = False
    bottleneck_pid_file = logs_dir / "bottleneck.pid"
    if bottleneck_pid_file.exists():
        try:
            pid = int(bottleneck_pid_file.read_text().strip())
            os.kill(pid, 0)
            bottleneck_running = True
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    if not bottleneck_running:
        try:
            import subprocess
            result = subprocess.run(
                ["tmux", "has-session", "-t", "bottleneck"],
                capture_output=True, timeout=5,
            )
            bottleneck_running = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Find bottleneck result files
    bottleneck_jsonl_files = sorted(two_bit_dir.glob("*_bottleneck.jsonl")) if two_bit_dir.exists() else []
    bottleneck_summaries = sorted(two_bit_dir.glob("*_bottleneck_summary.json")) if two_bit_dir.exists() else []
    codebook_files = sorted(two_bit_dir.glob("*_codebook_analysis.json")) if two_bit_dir.exists() else []

    has_bottleneck = bottleneck_jsonl_files or bottleneck_summaries or bottleneck_running

    if has_bottleneck:
        bn_status = "\033[92mRUNNING\033[0m" if bottleneck_running else "\033[91mSTOPPED\033[0m"
        print()
        print(f"  Bottleneck Optimization  [{bn_status}]")

        for bn_path in bottleneck_jsonl_files:
            bench_name = bn_path.stem.replace("_bottleneck", "")
            bench_display = BENCHMARK_NAMES.get(bench_name, bench_name)

            done = 0
            correct = 0
            correct_before = 0
            nan_count = 0
            total_loss_init = 0
            total_loss_final = 0
            valid_loss_count = 0
            total_time = 0
            expected = 50
            num_tokens = 1
            lm_decode_samples = []  # collect a few for display

            with open(bn_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                        done += 1
                        if r.get("correct"):
                            correct += 1
                        if r.get("initial_correct"):
                            correct_before += 1
                        if r.get("nan_detected"):
                            nan_count += 1
                        total_loss_init += r.get("initial_loss", 0)
                        if r.get("final_loss") is not None:
                            total_loss_final += r["final_loss"]
                            valid_loss_count += 1
                        total_time += r.get("optimization_time_s", 0)
                        num_tokens = r.get("num_tokens", 1)
                        # Collect last few LM head decodings for display
                        if r.get("final_lm_decode") and len(lm_decode_samples) < 3:
                            gt = r.get("ground_truth", "?")
                            top_word = r["final_lm_decode"][0][0][0] if r["final_lm_decode"][0] else "?"
                            lm_decode_samples.append((gt, top_word, r.get("correct", False)))
                    except json.JSONDecodeError:
                        continue

            # Check summary for expected count
            summary_path = two_bit_dir / f"{bench_name}_bottleneck_summary.json"
            num_image_tokens = None
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary = json.loads(f.read())
                    expected = summary.get("num_samples", expected)
                    num_image_tokens = summary.get("num_image_tokens")
                except (json.JSONDecodeError, KeyError):
                    pass

            last_update = datetime.fromtimestamp(bn_path.stat().st_mtime) if done > 0 else None

            acc_before = (correct_before / done * 100) if done > 0 else 0
            acc_after = (correct / done * 100) if done > 0 else 0
            avg_loss_init = (total_loss_init / done) if done > 0 else 0
            avg_loss_final = (total_loss_final / valid_loss_count) if valid_loss_count > 0 else 0
            avg_drop = avg_loss_init - avg_loss_final if valid_loss_count > 0 else 0
            avg_time = (total_time / done) if done > 0 else 0
            remaining = max(0, expected - done)
            eta_s = remaining * avg_time if done > 0 else 0

            if done >= expected and expected > 0:
                status = "\033[92m DONE \033[0m"
            elif done > 0:
                status = "\033[93m >>>  \033[0m"
            else:
                status = "      "

            # Compact info line
            expand_str = f" -> {num_image_tokens} positions" if num_image_tokens else ""
            print(f"  {bench_display}: {num_tokens} token{expand_str}")

            # Progress bar
            pct = (done / expected * 100) if expected > 0 else 0
            bar_width = 25
            filled = int(bar_width * pct / 100)
            bar = "=" * filled + (">" if filled < bar_width else "") + " " * max(0, bar_width - filled - 1)

            eta_str = format_duration(eta_s) if done > 0 and done < expected else ("done" if done >= expected else "-")
            nan_str = f"  NaN: {nan_count}" if nan_count > 0 else ""

            print(
                f"{status} [{bar}] {done}/{expected} ({pct:.0f}%)  "
                f"acc: {acc_before:.0f}% -> {acc_after:.0f}%  "
                f"loss: {avg_loss_init:.3f} -> {avg_loss_final:.3f} ({avg_drop:+.3f})  "
                f"ETA: {eta_str}{nan_str}"
            )

            if done > 0:
                print(
                    f"       avg {avg_time:.1f}s/question  "
                    f"updated {format_time_ago(last_update)}"
                )

            # Show a few recent LM head decodings (what the token looks like to the LLM)
            if lm_decode_samples:
                decode_strs = []
                for gt, top_word, was_correct in lm_decode_samples:
                    mark = "+" if was_correct else "-"
                    decode_strs.append(f"gt={gt}->'{top_word}'({mark})")
                print(f"       recent decodings: {', '.join(decode_strs)}")

        # Show codebook analysis results if available
        for cb_path in codebook_files:
            bench_name = cb_path.stem.replace("_codebook_analysis", "")
            bench_display = BENCHMARK_NAMES.get(bench_name, bench_name)
            try:
                with open(cb_path) as f:
                    cb = json.loads(f.read())

                print(f"\n  Codebook ({bench_display}):")

                # Per-class summary
                if cb.get("answer_classes"):
                    parts = []
                    for ans, stats in sorted(cb["answer_classes"].items()):
                        parts.append(f"{ans}: n={stats['count']} acc={stats['accuracy']:.0f}% spread={stats['within_class_spread_mean']:.2f}")
                    for p in parts:
                        print(f"    {p}")

                # Between-class distances
                if cb.get("between_class_distances"):
                    dist_parts = []
                    for pair, d in sorted(cb["between_class_distances"].items()):
                        dist_parts.append(f"{pair}: cos={d['cosine_similarity']:.3f}")
                    print(f"    distances: {', '.join(dist_parts)}")

                # Step-accuracy curve (just first and last)
                if cb.get("step_accuracy_curve"):
                    curve = cb["step_accuracy_curve"]
                    if len(curve) >= 2:
                        print(f"    accuracy: step {curve[0]['step']} = {curve[0]['accuracy']:.1f}%  ->  step {curve[-1]['step']} = {curve[-1]['accuracy']:.1f}%")

                # LM head decoded centroids
                if cb.get("lm_head_decoding"):
                    decode_parts = []
                    for ans, toks in sorted(cb["lm_head_decoding"].items()):
                        if toks and toks[0]:
                            top = toks[0][0]
                            decode_parts.append(f"{ans}->'{top['token']}'({top['prob']:.2f})")
                    if decode_parts:
                        print(f"    centroids: {', '.join(decode_parts)}")

            except (json.JSONDecodeError, KeyError):
                pass

        # Log hint
        bottleneck_log = logs_dir / "optimize_bottleneck.log"
        if bottleneck_log.exists():
            print(f"  Log: tail -f {bottleneck_log}")

        # In-progress detection from log tail (when no results file exists yet)
        if bottleneck_running and not bottleneck_jsonl_files:
            bottleneck_log = logs_dir / "optimize_bottleneck.log"
            if bottleneck_log.exists():
                try:
                    with open(bottleneck_log, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 4096))
                        tail = f.read().decode("utf-8", errors="ignore")
                    for line in reversed(tail.splitlines()):
                        if "Loading model" in line:
                            print(f"  (Loading model...)")
                            break
                        if "Discovered image token count" in line:
                            count = line.split("count: ")[-1].strip()
                            print(f"  (Discovered {count} image tokens, starting optimization...)")
                            break
                        if "BOTTLENECK per-question" in line:
                            print(f"  (Starting per-question optimization...)")
                            break
                except OSError:
                    pass

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
