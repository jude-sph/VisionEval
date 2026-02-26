"""Optimize universal bottleneck tokens — one shared token per answer class.

Tests whether a single embedding can steer the model to answer "A" (or B, C, D, E)
regardless of the question content.

Usage:
    # Quick pilot: 10 questions per class, 10 steps (~38 min on 3090)
    python scripts/optimize_universal.py

    # Full run
    python scripts/optimize_universal.py --max_samples_per_class 50 --num_steps 20
"""

import os
import sys
import logging
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("optimize_universal")


def main(
    benchmark: str = "mmmu",
    gpu_id: int = 0,
    max_samples_per_class: int = 10,
    num_tokens: int = 1,
    num_steps: int = 10,
    lr: float = 0.01,
    train_expand: int = 1,
    snapshot_every: int = 2,
    model_path: str = "nyu-visionx/cambrian-8b",
    conv_mode: str = "llama_3",
    results_dir: str = "results/universal",
    answer_classes: str = "A,B,C,D,E",
):
    """Run universal bottleneck optimisation.

    Args:
        benchmark: Benchmark name.
        gpu_id: Single GPU index.
        max_samples_per_class: Questions per answer class.
        num_tokens: Bottleneck tokens per class.
        num_steps: Gradient descent steps.
        lr: Adam learning rate.
        train_expand: Token copies for forward pass.
        snapshot_every: Evaluate accuracy every N steps.
        model_path: HuggingFace model path.
        conv_mode: Conversation template.
        results_dir: Where to save results.
        answer_classes: Comma-separated answer classes to optimize.
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "optimize_universal.log")),
        ],
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    classes = [c.strip() for c in answer_classes.split(",")]

    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"GPU: {gpu_id}")
    logger.info(f"Answer classes: {classes}")
    logger.info(f"Samples per class: {max_samples_per_class}")
    logger.info(f"Steps: {num_steps}, lr={lr}")
    logger.info(f"Results dir: {results_dir}")

    # Load model
    from src.model.loader import load_cambrian

    logger.info("Loading model...")
    tokenizer, model, image_processor, context_len = load_cambrian(
        model_path=model_path,
        gpu_ids=[0],
    )
    logger.info("Model loaded")

    # Load benchmark — need enough samples to fill all classes
    from src.benchmarks import get_benchmark

    total_needed = max_samples_per_class * len(classes) * 3  # overshoot to ensure coverage
    bench = get_benchmark(benchmark)
    bench.load(max_samples=total_needed)
    logger.info(f"Loaded {len(bench)} samples from {benchmark}")

    # Run universal optimisation
    from src.optimization.bottleneck_optimizer import optimize_bottleneck_universal

    logger.info("=" * 60)
    logger.info("PHASE: Universal bottleneck optimisation")
    logger.info("=" * 60)

    summary = optimize_bottleneck_universal(
        model=model,
        tokenizer=tokenizer,
        benchmark=bench,
        image_processor=image_processor,
        num_tokens=num_tokens,
        max_samples_per_class=max_samples_per_class,
        num_steps=num_steps,
        lr=lr,
        train_expand=train_expand,
        conv_mode=conv_mode,
        results_dir=results_dir,
        snapshot_every=snapshot_every,
        answer_classes=classes,
    )

    print("\n" + "=" * 60)
    print("UNIVERSAL BOTTLENECK RESULTS")
    print("=" * 60)
    print(f"Overall accuracy: {summary['overall_accuracy']}% "
          f"({summary['total_correct']}/{summary['total_questions']})")
    print(f"Total time: {summary['total_time_s']:.0f}s")
    print()
    for ans, stats in summary['per_class'].items():
        print(f"  {ans}: {stats['final_accuracy']}% "
              f"({stats['n_questions']} questions), "
              f"loss={stats['final_avg_loss']:.3f}, "
              f"{stats['time_s']:.0f}s")

    logger.info("All done!")


if __name__ == "__main__":
    fire.Fire(main)
