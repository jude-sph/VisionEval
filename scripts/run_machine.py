"""Launch evaluation jobs for assigned benchmarks on the current machine.

Usage:
    python scripts/run_machine.py              # Prompts which machine you're on
    python scripts/run_machine.py --machine A  # Skip prompt, force Machine A (4x Pascal)
    python scripts/run_machine.py --machine B  # Skip prompt, force Machine B (3090)
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

# Benchmark assignments per machine (from plan)
MACHINE_B_JOBS = [
    # (benchmark, condition, gpu_ids, load_8bit)
    ("gqa", "normal", "0", False),
    ("gqa", "no_image", "0", False),
    ("gqa", "wrong_image", "0", False),
    ("gqa", "gaussian_noise", "0", False),
    ("mmmu", "normal", "0", False),
    ("mmmu", "no_image", "0", False),
    ("mmmu", "wrong_image", "0", False),
    ("mmmu", "gaussian_noise", "0", False),
]

# Fallback: INT8 failed on Pascal (compute 6.1), use FP16 across all 4 GPUs
# Single instance, sequential jobs (can't split — all 4 GPUs needed for one model)
MACHINE_A_JOBS = [
    # (benchmark, condition, gpu_ids, load_8bit)
    ("mmbench", "normal", "0,1,2,3", False),
    ("mmbench", "no_image", "0,1,2,3", False),
    ("mmbench", "wrong_image", "0,1,2,3", False),
    ("mmbench", "gaussian_noise", "0,1,2,3", False),
    ("pope", "normal", "0,1,2,3", False),
    ("pope", "no_image", "0,1,2,3", False),
    ("pope", "wrong_image", "0,1,2,3", False),
    ("pope", "gaussian_noise", "0,1,2,3", False),
    ("textvqa", "normal", "0,1,2,3", False),
    ("textvqa", "no_image", "0,1,2,3", False),
    ("textvqa", "wrong_image", "0,1,2,3", False),
    ("textvqa", "gaussian_noise", "0,1,2,3", False),
    ("scienceqa", "normal", "0,1,2,3", False),
    ("scienceqa", "no_image", "0,1,2,3", False),
    ("scienceqa", "wrong_image", "0,1,2,3", False),
    ("scienceqa", "gaussian_noise", "0,1,2,3", False),
]


def get_gpu_summary() -> str:
    """Get a human-readable summary of GPUs on this machine."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "(could not query GPUs)"


def prompt_machine() -> str:
    """Ask the user which machine they're on, showing detected GPUs for context."""
    gpu_info = get_gpu_summary()

    print()
    print("=" * 60)
    print("  Which machine is this?")
    print("=" * 60)
    print()
    print("  Detected GPUs:")
    for line in gpu_info.splitlines():
        print(f"    {line.strip()}")
    print()
    print("  A) Machine A — 4x Titan X Pascal (compute 6.1)")
    print("     Runs: MMBench, POPE, TextVQA, ScienceQA")
    print("     Config: FP16, 4-way tensor parallel, 1 instance")
    print()
    print("  B) Machine B — 1x RTX 3090 (+ unusable Titan X Maxwell)")
    print("     Runs: GQA, MMMU")
    print("     Config: FP16, single GPU")
    print()

    while True:
        choice = input("  Enter A or B: ").strip().upper()
        if choice in ("A", "B"):
            return choice
        print("  Please enter A or B.")


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
        if max_samples:
            cmd.extend(["--max_samples", str(max_samples)])

        logger.info(f"Starting: {benchmark}/{condition} on GPUs {gpu_ids}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode != 0:
            logger.error(f"Failed: {benchmark}/{condition} (exit code {result.returncode})")
        else:
            logger.info(f"Completed: {benchmark}/{condition}")


def main(
    machine: str | None = None,
    max_samples: int | None = None,
    dry_run: bool = False,
):
    """Run assigned evaluation jobs on this machine.

    Args:
        machine: Machine identity ("A" or "B"). Prompts interactively if not given.
        max_samples: Limit samples per benchmark (for testing).
        dry_run: Print jobs without running them.
    """
    if machine is None:
        machine = prompt_machine()
    else:
        machine = machine.upper()
    logger.info(f"Machine: {machine}")

    if machine == "B":
        jobs = MACHINE_B_JOBS
        logger.info(f"Machine B (3090): {len(jobs)} jobs, single GPU FP16")
        if dry_run:
            for b, c, g, q in jobs:
                print(f"  {b}/{c} on GPU {g} (INT8={q})")
            return
        run_jobs_sequential(jobs, max_samples)

    elif machine == "A":
        jobs = MACHINE_A_JOBS
        logger.info(f"Machine A (Pascal): {len(jobs)} jobs, 4-way FP16 tensor parallel")
        if dry_run:
            for b, c, g, q in jobs:
                print(f"  {b}/{c} on GPUs {g} (INT8={q})")
            return
        run_jobs_sequential(jobs, max_samples)
    else:
        logger.error(f"Unknown machine: {machine}. Use --machine A or --machine B")
        sys.exit(1)

    logger.info("All evaluation jobs completed!")


if __name__ == "__main__":
    fire.Fire(main)
