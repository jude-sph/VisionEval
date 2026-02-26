"""Optimize a SINGLE token shared across ALL questions regardless of answer class.

Tests the zero-bit hypothesis: can one generic "activate" signal make the model
answer every question correctly without encoding any answer information?

Usage:
    python scripts/optimize_single_token.py
    python scripts/optimize_single_token.py --max_samples 100 --num_steps 40
"""

import os
import sys
import logging
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("optimize_single_token")


def main(
    benchmark: str = "mmmu",
    gpu_id: int = 0,
    max_samples: int = 50,
    num_tokens: int = 1,
    num_steps: int = 30,
    lr: float = 0.01,
    train_expand: int = 1,
    snapshot_every: int = 5,
    model_path: str = "nyu-visionx/cambrian-8b",
    conv_mode: str = "llama_3",
    results_dir: str = "results/single_token",
):
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "optimize_single_token.log")),
        ],
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Steps: {num_steps}, lr={lr}")
    logger.info(f"Results dir: {results_dir}")

    from src.model.loader import load_cambrian

    logger.info("Loading model...")
    tokenizer, model, image_processor, context_len = load_cambrian(
        model_path=model_path,
        gpu_ids=[0],
    )
    logger.info("Model loaded")

    from src.benchmarks import get_benchmark

    bench = get_benchmark(benchmark)
    bench.load(max_samples=max_samples * 3)  # overshoot to ensure enough MCQ
    logger.info(f"Loaded {len(bench)} samples from {benchmark}")

    from src.optimization.bottleneck_optimizer import optimize_bottleneck_single

    logger.info("=" * 60)
    logger.info("PHASE: Single-token experiment (zero-bit hypothesis)")
    logger.info("=" * 60)

    summary = optimize_bottleneck_single(
        model=model,
        tokenizer=tokenizer,
        benchmark=bench,
        image_processor=image_processor,
        num_tokens=num_tokens,
        max_samples=max_samples,
        num_steps=num_steps,
        lr=lr,
        train_expand=train_expand,
        conv_mode=conv_mode,
        results_dir=results_dir,
        snapshot_every=snapshot_every,
    )

    print("\n" + "=" * 60)
    print("SINGLE TOKEN RESULTS")
    print("=" * 60)
    print(f"Overall accuracy: {summary['final_accuracy']}% "
          f"({summary['n_questions']} questions)")
    print(f"Total time: {summary['total_time_s']:.0f}s")
    print()
    for ans, stats in summary['per_class_accuracy'].items():
        print(f"  {ans}: {stats['accuracy']}% ({stats['n_correct']}/{stats['n_questions']})")

    logger.info("All done!")


if __name__ == "__main__":
    fire.Fire(main)
