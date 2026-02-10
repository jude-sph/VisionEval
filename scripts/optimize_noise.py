"""Optimize noise embeddings to maximize benchmark accuracy.

Finds embedding-space inputs that maximize P(correct answer) for Cambrian-8B,
without using any real images.

Usage:
    # Both modes (universal first, then per-question) — default
    python scripts/optimize_noise.py --benchmark mmmu --max_samples 50

    # Universal only
    python scripts/optimize_noise.py --mode universal --num_epochs 10

    # Per-question only
    python scripts/optimize_noise.py --mode per_question --num_steps 20

    # Smoke test
    python scripts/optimize_noise.py --max_samples 1 --mode per_question --num_steps 5
"""

import os
import sys
import logging
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("optimize_noise")


def main(
    benchmark: str = "mmmu",
    gpu_ids: str = "0,1,2,3",
    max_samples: int = 50,
    mode: str = "both",
    num_steps: int = 20,
    num_epochs: int = 10,
    lr: float = 0.001,
    model_path: str = "nyu-visionx/cambrian-8b",
    conv_mode: str = "llama_3",
    results_dir: str = "results/optimization",
):
    """Run embedding-space noise optimization on a benchmark.

    Args:
        benchmark: Benchmark name (mmmu, pope, etc.).
        gpu_ids: Comma-separated GPU indices (default: 0,1,2,3 for 4x Titan X Pascal).
        max_samples: Number of questions to optimize.
        mode: 'universal', 'per_question', or 'both' (universal first, then per-question).
        num_steps: Gradient descent steps per question (per-question mode).
        num_epochs: Number of epochs (universal mode).
        lr: Adam learning rate for optimization.
        model_path: HuggingFace model path.
        conv_mode: Conversation template.
        results_dir: Directory for optimization results.
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "optimize_noise.log")),
        ],
    )

    # Parse GPU IDs
    if isinstance(gpu_ids, (list, tuple)):
        gpu_list = [int(x) for x in gpu_ids]
    else:
        gpu_list = [int(x) for x in str(gpu_ids).split(",")]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    remapped_gpus = list(range(len(gpu_list)))

    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Mode: {mode}")
    logger.info(f"GPUs: {gpu_list} (remapped to {remapped_gpus})")
    logger.info(f"Max samples: {max_samples}")
    if mode in ("universal", "both"):
        logger.info(f"Universal epochs: {num_epochs}")
    if mode in ("per_question", "both"):
        logger.info(f"Per-question steps: {num_steps}")
    logger.info(f"Learning rate: {lr}")

    # Load model
    from src.model.loader import load_cambrian

    logger.info(f"Loading model across {len(remapped_gpus)} GPU(s)...")
    tokenizer, model, image_processor, context_len = load_cambrian(
        model_path=model_path,
        gpu_ids=remapped_gpus,
    )
    logger.info("Model loaded successfully")

    # Load benchmark
    from src.benchmarks import get_benchmark

    bench = get_benchmark(benchmark)
    bench.load(max_samples=max_samples)

    # Run optimization(s)
    from src.optimization.embedding_optimizer import optimize_universal, optimize_per_question

    if mode in ("universal", "both"):
        logger.info("=" * 60)
        logger.info("PHASE 1: Universal embedding optimization")
        logger.info("=" * 60)
        universal_summary = optimize_universal(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            benchmark=bench,
            max_samples=max_samples,
            num_epochs=num_epochs,
            lr=lr,
            conv_mode=conv_mode,
            results_dir=results_dir,
        )
        logger.info(f"Universal results: {universal_summary}")

    if mode in ("per_question", "both"):
        logger.info("=" * 60)
        logger.info("PHASE 2: Per-question embedding optimization")
        logger.info("=" * 60)
        per_q_summary = optimize_per_question(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            benchmark=bench,
            max_samples=max_samples,
            num_steps=num_steps,
            lr=lr,
            conv_mode=conv_mode,
            results_dir=results_dir,
            _skip_setup=mode == "both",  # already set up by universal
        )
        logger.info(f"Per-question results: {per_q_summary}")

    logger.info("All optimization complete!")


if __name__ == "__main__":
    fire.Fire(main)
