"""Launch all evaluation jobs sequentially on single GPU (3090 FP16).

Usage:
    python scripts/run_machine.py                  # Run all benchmarks x conditions
    python scripts/run_machine.py --max_samples 10 # Quick test with 10 samples each
    python scripts/run_machine.py --dry_run         # Print jobs without running
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_machine")

# Per-benchmark sample limits (None = use full dataset)
# GQA has ~12k instructions but we cap at 5000 for consistent timing/stats
BENCHMARK_SAMPLE_LIMITS = {
    "gqa": 5000,
}

# Expected sample counts per benchmark (full dataset sizes)
EXPECTED_SAMPLES = {
    "mmmu": 900,
    "mmbench": 4329,
    "pope": 9000,
    "textvqa": 3000,
    "gqa": 5000,
    "scienceqa": 2000,
}

# All benchmarks + noise optimization run FP16 on single GPU (3090).
# INT8 is incompatible with Cambrian/accelerate version on this machine.
# Noise optimization fits in 24GB by offloading vision encoders to CPU
# (they're bypassed by encode_images_hook) + gradient checkpointing.
ALL_JOBS = [
    # (benchmark, condition, gpu_ids, load_8bit)
    ("mmmu", "normal", "0", False),
    ("mmmu", "no_image", "0", False),
    ("mmmu", "wrong_image", "0", False),
    ("mmmu", "gaussian_noise", "0", False),
    ("mmbench", "normal", "0", False),
    ("mmbench", "no_image", "0", False),
    ("mmbench", "wrong_image", "0", False),
    ("mmbench", "gaussian_noise", "0", False),
    ("pope", "normal", "0", False),
    ("pope", "no_image", "0", False),
    ("pope", "wrong_image", "0", False),
    ("pope", "gaussian_noise", "0", False),
    ("textvqa", "normal", "0", False),
    ("textvqa", "no_image", "0", False),
    ("textvqa", "wrong_image", "0", False),
    ("textvqa", "gaussian_noise", "0", False),
    # Temporarily disabled — add back if needed:
    # ("gqa", "normal", "0", False),
    # ("gqa", "no_image", "0", False),
    # ("gqa", "wrong_image", "0", False),
    # ("gqa", "gaussian_noise", "0", False),
    # ("scienceqa", "normal", "0", False),
    # ("scienceqa", "no_image", "0", False),
    # ("scienceqa", "wrong_image", "0", False),
    # ("scienceqa", "gaussian_noise", "0", False),
]


def _count_results(results_dir: str, benchmark: str, condition: str) -> int:
    """Count completed results in JSONL file without loading into memory."""
    path = Path(results_dir) / "raw" / f"{benchmark}_{condition}.jsonl"
    if not path.exists():
        return 0
    count = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def run_jobs_sequential(jobs: list[tuple], max_samples: int | None = None, results_dir: str = "results"):
    """Run a list of (benchmark, condition, gpu_ids, load_8bit) jobs sequentially."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_single.py")

    for benchmark, condition, gpu_ids, load_8bit in jobs:
        # Check if this job is already complete before loading the model
        effective_max = max_samples or BENCHMARK_SAMPLE_LIMITS.get(benchmark)
        expected = effective_max or EXPECTED_SAMPLES.get(benchmark, 0)
        if expected > 0:
            done = _count_results(results_dir, benchmark, condition)
            if done >= expected:
                logger.info(f"Skipping: {benchmark}/{condition} — already complete ({done}/{expected})")
                continue

        cmd = [
            sys.executable, script,
            "--benchmark", benchmark,
            "--condition", condition,
            "--gpu_ids", gpu_ids,
        ]
        if load_8bit:
            cmd.append("--load_8bit")

        if effective_max:
            cmd.extend(["--max_samples", str(effective_max)])

        logger.info(f"Starting: {benchmark}/{condition} on GPUs {gpu_ids}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode != 0:
            logger.error(f"Failed: {benchmark}/{condition} (exit code {result.returncode})")
        else:
            logger.info(f"Completed: {benchmark}/{condition}")


def main(
    max_samples: int | None = None,
    dry_run: bool = False,
    results_dir: str = "results",
):
    """Run all evaluation jobs sequentially on single GPU.

    Args:
        max_samples: Limit samples per benchmark (for testing).
        dry_run: Print jobs without running them.
        results_dir: Directory for results.
    """
    jobs = ALL_JOBS
    logger.info(f"Running {len(jobs)} jobs sequentially on single GPU FP16")

    if dry_run:
        for b, c, g, q in jobs:
            effective_max = max_samples or BENCHMARK_SAMPLE_LIMITS.get(b)
            expected = effective_max or EXPECTED_SAMPLES.get(b, 0)
            done = _count_results(results_dir, b, c)
            skip = " [SKIP — complete]" if expected > 0 and done >= expected else ""
            print(f"  {b}/{c} on GPU {g} (INT8={q}) ({done}/{expected}){skip}")
        return

    run_jobs_sequential(jobs, max_samples, results_dir)
    logger.info("All evaluation jobs completed!")


if __name__ == "__main__":
    fire.Fire(main)
