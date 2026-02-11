"""Optimize noise embeddings/pixels to maximize benchmark accuracy.

Finds inputs that maximize P(correct answer) for Cambrian-8B,
without using any real images.

Embedding modes optimize encoder output tensors (bypassing vision encoders).
Pixel modes optimize raw pixel values through the real vision encoders,
producing visible noise images.

Usage:
    # Embedding optimization (both universal + per-question) — default
    python scripts/optimize_noise.py --benchmark mmmu --max_samples 50

    # Pixel optimization (both universal + per-question)
    python scripts/optimize_noise.py --mode pixel_both --max_samples 50

    # All modes (embedding + pixel)
    python scripts/optimize_noise.py --mode all --max_samples 50

    # Pixel per-question only
    python scripts/optimize_noise.py --mode pixel_per_question --pixel_steps 20

    # Smoke test
    python scripts/optimize_noise.py --max_samples 1 --mode pixel_per_question --pixel_steps 5
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
    pixel_steps: int = 20,
    pixel_epochs: int = 10,
    pixel_lr: float = 0.01,
    model_path: str = "nyu-visionx/cambrian-8b",
    conv_mode: str = "llama_3",
    results_dir: str = "results/optimization",
    resume: bool = False,
    batch_size: int = 1,
    train_ratio: float = 1.0,
    patience: int = 0,
    seed: int = 42,
    eval_every: int = 1,
):
    """Run noise optimization on a benchmark.

    Args:
        benchmark: Benchmark name (mmmu, pope, etc.).
        gpu_ids: Comma-separated GPU indices (default: 0,1,2,3 for 4x Titan X Pascal).
        max_samples: Number of questions to optimize.
        mode: Optimization mode:
            'universal'        - Embedding universal only
            'per_question'     - Embedding per-question only
            'both'             - Embedding universal + per-question (default)
            'pixel_universal'  - Pixel universal only
            'pixel_per_question' - Pixel per-question only
            'pixel_both'       - Pixel universal + per-question
            'all'              - All four modes (embedding both + pixel both)
        num_steps: Gradient descent steps per question (embedding per-question).
        num_epochs: Number of epochs (embedding universal).
        lr: Adam learning rate for embedding optimization.
        pixel_steps: Gradient descent steps per question (pixel per-question).
        pixel_epochs: Number of epochs (pixel universal).
        pixel_lr: Adam learning rate for pixel optimization (higher default
            because gradients attenuate through vision encoders).
        model_path: HuggingFace model path.
        conv_mode: Conversation template.
        results_dir: Directory for optimization results.
        resume: Resume universal optimization from saved checkpoint tensors.
        batch_size: Minibatch size for universal optimization (1 = single-sample).
        train_ratio: Fraction of samples for training (rest for test). 1.0 = no split.
        patience: Early stopping patience on test accuracy (0 = disabled).
        seed: Random seed for reproducible train/test split.
        eval_every: Evaluate on test set every N epochs.
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

    # Determine which phases to run
    run_emb_universal = mode in ("universal", "both", "all")
    run_emb_per_q = mode in ("per_question", "both", "all")
    run_pix_universal = mode in ("pixel_universal", "pixel_both", "all")
    run_pix_per_q = mode in ("pixel_per_question", "pixel_both", "all")

    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Mode: {mode}")
    logger.info(f"GPUs: {gpu_list} (remapped to {remapped_gpus})")
    logger.info(f"Max samples: {max_samples}")
    if run_emb_universal:
        logger.info(f"Embedding universal: {num_epochs} epochs, lr={lr}, batch_size={batch_size}")
        if train_ratio < 1.0:
            logger.info(f"  Train/test split: {train_ratio:.0%} train, seed={seed}")
        if patience > 0:
            logger.info(f"  Early stopping: patience={patience}, eval_every={eval_every}")
    if run_emb_per_q:
        logger.info(f"Embedding per-question: {num_steps} steps, lr={lr}")
    if run_pix_universal:
        logger.info(f"Pixel universal: {pixel_epochs} epochs, lr={pixel_lr}")
    if run_pix_per_q:
        logger.info(f"Pixel per-question: {pixel_steps} steps, lr={pixel_lr}")

    # Load model
    from src.model.loader import load_cambrian

    # Pixel optimization keeps vision encoders on GPU 0 (~3.8GB) and needs
    # room for backward activations. Reduce GPU 0's LLM allocation from
    # 7GiB to 4GiB so other GPUs absorb the extra layers.
    needs_pixel = run_pix_universal or run_pix_per_q
    gpu0_mem = "4GiB" if needs_pixel else "7GiB"

    logger.info(f"Loading model across {len(remapped_gpus)} GPU(s) (GPU 0: {gpu0_mem})...")
    tokenizer, model, image_processor, context_len = load_cambrian(
        model_path=model_path,
        gpu_ids=remapped_gpus,
        gpu0_max_memory=gpu0_mem,
    )
    logger.info("Model loaded successfully")

    # Load benchmark
    from src.benchmarks import get_benchmark

    bench = get_benchmark(benchmark)
    bench.load(max_samples=max_samples)

    # === EMBEDDING OPTIMIZATION ===
    if run_emb_universal or run_emb_per_q:
        from src.optimization.embedding_optimizer import optimize_universal, optimize_per_question

        if run_emb_universal:
            logger.info("=" * 60)
            logger.info("PHASE: Embedding universal optimization")
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
                resume=resume,
                batch_size=batch_size,
                train_ratio=train_ratio,
                patience=patience,
                seed=seed,
                eval_every=eval_every,
            )
            logger.info(f"Embedding universal results: {universal_summary}")

        if run_emb_per_q:
            logger.info("=" * 60)
            logger.info("PHASE: Embedding per-question optimization")
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
                _skip_setup=run_emb_universal,
            )
            logger.info(f"Embedding per-question results: {per_q_summary}")

    # === PIXEL OPTIMIZATION ===
    if run_pix_universal or run_pix_per_q:
        from src.optimization.pixel_optimizer import optimize_pixel_universal, optimize_pixel_per_question

        # Vision encoders need to be back on GPU for pixel optimization.
        # If embedding optimization offloaded them, move them back.
        if run_emb_universal or run_emb_per_q:
            inner = getattr(model, "model", model)
            towers = getattr(inner, "vision_tower_aux_list", None)
            if towers:
                device = next(model.parameters()).device
                for tower in towers:
                    tower.to(device)
                logger.info(f"Moved {len(towers)} vision encoders back to GPU")
                import torch
                torch.cuda.empty_cache()

        if run_pix_universal:
            logger.info("=" * 60)
            logger.info("PHASE: Pixel universal optimization")
            logger.info("=" * 60)
            pix_universal_summary = optimize_pixel_universal(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                benchmark=bench,
                max_samples=max_samples,
                num_epochs=pixel_epochs,
                lr=pixel_lr,
                conv_mode=conv_mode,
                results_dir=results_dir,
            )
            logger.info(f"Pixel universal results: {pix_universal_summary}")

        if run_pix_per_q:
            logger.info("=" * 60)
            logger.info("PHASE: Pixel per-question optimization")
            logger.info("=" * 60)
            pix_per_q_summary = optimize_pixel_per_question(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                benchmark=bench,
                max_samples=max_samples,
                num_steps=pixel_steps,
                lr=pixel_lr,
                conv_mode=conv_mode,
                results_dir=results_dir,
                _skip_setup=run_pix_universal,
            )
            logger.info(f"Pixel per-question results: {pix_per_q_summary}")

    logger.info("All optimization complete!")


if __name__ == "__main__":
    fire.Fire(main)
