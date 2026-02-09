"""Auto-detect machine and launch evaluation jobs for assigned benchmarks.

Usage:
    python scripts/run_machine.py              # Auto-detect
    python scripts/run_machine.py --machine A  # Force Machine A (Pascal)
    python scripts/run_machine.py --machine B  # Force Machine B (3090)
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
    ("mmmu", "normal", "0", False),
    ("mmmu", "no_image", "0", False),
    ("mmmu", "wrong_image", "0", False),
]

MACHINE_A_INSTANCE_1 = [
    ("mmbench", "normal", "0,1", True),
    ("mmbench", "no_image", "0,1", True),
    ("mmbench", "wrong_image", "0,1", True),
    ("pope", "normal", "0,1", True),
    ("pope", "no_image", "0,1", True),
    ("pope", "wrong_image", "0,1", True),
]

MACHINE_A_INSTANCE_2 = [
    ("textvqa", "normal", "2,3", True),
    ("textvqa", "no_image", "2,3", True),
    ("textvqa", "wrong_image", "2,3", True),
    ("scienceqa", "normal", "2,3", True),
    ("scienceqa", "no_image", "2,3", True),
    ("scienceqa", "wrong_image", "2,3", True),
]


def detect_machine() -> str:
    """Auto-detect which machine we're on based on GPU names."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        gpu_names = result.stdout.strip()
        if "3090" in gpu_names:
            return "B"
        if "TITAN X" in gpu_names or "TITAN Xp" in gpu_names:
            return "A"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


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
    """Detect machine and run assigned evaluation jobs.

    Args:
        machine: Force machine identity ("A" or "B"). Auto-detects if None.
        max_samples: Limit samples per benchmark (for testing).
        dry_run: Print jobs without running them.
    """
    if machine is None:
        machine = detect_machine()
        logger.info(f"Auto-detected: Machine {machine}")
    else:
        machine = machine.upper()
        logger.info(f"Forced: Machine {machine}")

    if machine == "B":
        jobs = MACHINE_B_JOBS
        logger.info(f"Machine B (3090): {len(jobs)} jobs, single GPU FP16")
        if dry_run:
            for b, c, g, q in jobs:
                print(f"  {b}/{c} on GPU {g} (INT8={q})")
            return
        run_jobs_sequential(jobs, max_samples)

    elif machine == "A":
        logger.info(f"Machine A (Pascal): launching 2 parallel instances")
        if dry_run:
            print("Instance 1:")
            for b, c, g, q in MACHINE_A_INSTANCE_1:
                print(f"  {b}/{c} on GPUs {g} (INT8={q})")
            print("Instance 2:")
            for b, c, g, q in MACHINE_A_INSTANCE_2:
                print(f"  {b}/{c} on GPUs {g} (INT8={q})")
            return

        # Launch two instances in parallel using subprocess
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_single.py")
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processes = []

        for instance_name, jobs in [("Instance 1", MACHINE_A_INSTANCE_1), ("Instance 2", MACHINE_A_INSTANCE_2)]:
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
                processes.append((instance_name, benchmark, condition, cmd, cwd))

        # Run instance 1 and instance 2 jobs interleaved to balance GPU usage
        # Each instance's jobs run sequentially within the instance,
        # but the two instances run in parallel
        import concurrent.futures

        def run_instance(instance_jobs):
            for inst_name, bench, cond, cmd, wd in instance_jobs:
                logger.info(f"[{inst_name}] Starting: {bench}/{cond}")
                result = subprocess.run(cmd, cwd=wd)
                if result.returncode != 0:
                    logger.error(f"[{inst_name}] Failed: {bench}/{cond}")
                else:
                    logger.info(f"[{inst_name}] Completed: {bench}/{cond}")

        # Split processes back into two instance groups
        inst1_procs = [(n, b, c, cmd, wd) for n, b, c, cmd, wd in processes if n == "Instance 1"]
        inst2_procs = [(n, b, c, cmd, wd) for n, b, c, cmd, wd in processes if n == "Instance 2"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(run_instance, inst1_procs)
            f2 = executor.submit(run_instance, inst2_procs)
            f1.result()
            f2.result()

        logger.info("All Machine A jobs completed")
    else:
        logger.error(f"Unknown machine: {machine}. Use --machine A or --machine B")
        sys.exit(1)

    logger.info("All evaluation jobs completed!")


if __name__ == "__main__":
    fire.Fire(main)
