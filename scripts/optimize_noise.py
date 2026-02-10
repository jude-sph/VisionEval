"""Optimize noise embeddings to maximize benchmark accuracy.

Finds embedding-space inputs that maximize P(correct answer) for Cambrian-8B,
without using any real images.

Usage:
    # 4x Titan X Pascal (default — dedicated noise optimization machine)
    python scripts/optimize_noise.py --benchmark mmmu --max_samples 50

    # Single GPU (e.g., 3090 — tight, may need gradient checkpointing)
    python scripts/optimize_noise.py --benchmark mmmu --gpu_ids 0 --max_samples 1

    # Smoke test
    python scripts/optimize_noise.py --benchmark mmmu --max_samples 1
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
    num_steps: int = 50,
    lr: float = 0.01,
    model_path: str = "nyu-visionx/cambrian-8b",
    conv_mode: str = "llama_3",
    results_dir: str = "results/optimization",
):
    """Run embedding-space noise optimization on a benchmark.

    Args:
        benchmark: Benchmark name (mmmu, pope, etc.).
        gpu_ids: Comma-separated GPU indices (default: 0,1,2,3 for 4x Titan X Pascal).
        max_samples: Number of questions to optimize.
        num_steps: Gradient descent steps per question.
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
    logger.info(f"GPUs: {gpu_list} (remapped to {remapped_gpus})")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Optimization steps: {num_steps}")
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

    # Run optimization
    from src.optimization.embedding_optimizer import optimize_per_question

    summary = optimize_per_question(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        benchmark=bench,
        max_samples=max_samples,
        num_steps=num_steps,
        lr=lr,
        conv_mode=conv_mode,
        results_dir=results_dir,
    )

    logger.info(f"Done! Results: {summary}")


if __name__ == "__main__":
    fire.Fire(main)
