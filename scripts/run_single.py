"""Run a single benchmark under a single condition on specified GPU(s).

Usage:
    python scripts/run_single.py --benchmark mmmu --condition normal --gpu_ids 0
    python scripts/run_single.py --benchmark pope --condition no_image --gpu_ids 0,1 --load_8bit
    python scripts/run_single.py --benchmark mmmu --condition normal --max_samples 10  # dev test
"""

import os
import sys
import logging
import fire

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("run_single")


def main(
    benchmark: str,
    condition: str = "normal",
    gpu_ids: str = "0",
    load_8bit: bool = False,
    max_samples: int | None = None,
    model_path: str = "nyu-visionx/cambrian-8b",
    conv_mode: str = "llama_3",
    max_new_tokens: int = 128,
    results_dir: str = "results",
):
    """Run evaluation for one benchmark + one condition.

    Args:
        benchmark: Benchmark name (mmmu, mmbench, scienceqa, pope, textvqa, gqa).
        condition: Condition name (normal, no_image, wrong_image).
        gpu_ids: Comma-separated GPU indices (e.g., "0" or "0,1").
        load_8bit: Use INT8 quantization.
        max_samples: Limit number of samples (for testing).
        model_path: HuggingFace model path.
        conv_mode: Conversation template.
        max_new_tokens: Max generation length.
        results_dir: Directory for results.
    """
    # Set up logging to both console and per-job log file
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{benchmark}_{condition}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    # fire may parse gpu_ids as int (0), tuple (0,1), or str ("0,1")
    if isinstance(gpu_ids, (list, tuple)):
        gpu_list = [int(x) for x in gpu_ids]
    else:
        gpu_list = [int(x) for x in str(gpu_ids).split(",")]
    gpu_ids = ",".join(str(g) for g in gpu_list)

    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # Remap to 0-indexed within the visible set
    remapped_gpus = list(range(len(gpu_list)))

    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Condition: {condition}")
    logger.info(f"GPUs: {gpu_list} (remapped to {remapped_gpus})")
    logger.info(f"INT8: {load_8bit}")
    logger.info(f"Max samples: {max_samples}")

    # Load model
    from src.model.loader import load_cambrian

    logger.info("Loading model...")
    tokenizer, model, image_processor, context_len = load_cambrian(
        model_path=model_path,
        gpu_ids=remapped_gpus,
        load_8bit=load_8bit,
    )
    logger.info("Model loaded successfully")

    # Get benchmark and condition
    from src.benchmarks import get_benchmark
    from src.image_conditions import get_condition
    from src.evaluation.runner import run_evaluation
    from src.evaluation.results_store import ResultsStore

    bench = get_benchmark(benchmark)
    cond = get_condition(condition)
    store = ResultsStore(results_dir)

    # Run evaluation
    metrics = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        benchmark=bench,
        condition=cond,
        results_store=store,
        max_samples=max_samples,
        conv_mode=conv_mode,
        max_new_tokens=max_new_tokens,
    )

    logger.info(f"Completed: {metrics}")


if __name__ == "__main__":
    fire.Fire(main)
