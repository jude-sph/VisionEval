"""Launch all evaluation jobs sequentially on single GPU (3090 INT8).

Usage:
    python scripts/run_machine.py                  # Run all benchmarks x conditions
    python scripts/run_machine.py --max_samples 10 # Quick test with 10 samples each
    python scripts/run_machine.py --dry_run         # Print jobs without running
"""

import os
import sys
import subprocess
import logging
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

# All benchmarks run on single GPU (3090 INT8) — multi-GPU tensor parallel
# is incompatible with Cambrian's multi-encoder architecture.
# INT8 quantization used for consistency with noise optimization (which
# requires backprop headroom). All conditions compared under same model.
ALL_JOBS = [
    # (benchmark, condition, gpu_ids, load_8bit)
    ("gqa", "normal", "0", True),
    ("gqa", "no_image", "0", True),
    ("gqa", "wrong_image", "0", True),
    ("gqa", "gaussian_noise", "0", True),
    ("mmmu", "normal", "0", True),
    ("mmmu", "no_image", "0", True),
    ("mmmu", "wrong_image", "0", True),
    ("mmmu", "gaussian_noise", "0", True),
    ("mmbench", "normal", "0", True),
    ("mmbench", "no_image", "0", True),
    ("mmbench", "wrong_image", "0", True),
    ("mmbench", "gaussian_noise", "0", True),
    ("pope", "normal", "0", True),
    ("pope", "no_image", "0", True),
    ("pope", "wrong_image", "0", True),
    ("pope", "gaussian_noise", "0", True),
    ("textvqa", "normal", "0", True),
    ("textvqa", "no_image", "0", True),
    ("textvqa", "wrong_image", "0", True),
    ("textvqa", "gaussian_noise", "0", True),
    ("scienceqa", "normal", "0", True),
    ("scienceqa", "no_image", "0", True),
    ("scienceqa", "wrong_image", "0", True),
    ("scienceqa", "gaussian_noise", "0", True),
]


def run_jobs_sequential(jobs: list[tuple], max_samples: int | None = None):
    """Run a list of (benchmark, condition, gpu_ids, load_8bit) jobs sequentially."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_single.py")

    for benchmark, condition, gpu_ids, load_8bit in jobs:
        cmd = [
            sys.executable, script,
            "--benchmark", benchmark,
            "--condition", condition,
            "--gpu_ids", gpu_ids,
        ]
        if load_8bit:
            cmd.append("--load_8bit")

        # Use CLI override, or per-benchmark limit, or no limit
        effective_max = max_samples or BENCHMARK_SAMPLE_LIMITS.get(benchmark)
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
):
    """Run all evaluation jobs sequentially on single GPU.

    Args:
        max_samples: Limit samples per benchmark (for testing).
        dry_run: Print jobs without running them.
    """
    jobs = ALL_JOBS
    logger.info(f"Running {len(jobs)} jobs sequentially on single GPU INT8")

    if dry_run:
        for b, c, g, q in jobs:
            print(f"  {b}/{c} on GPU {g} (INT8={q})")
        return

    run_jobs_sequential(jobs, max_samples)
    logger.info("All evaluation jobs completed!")


if __name__ == "__main__":
    fire.Fire(main)
