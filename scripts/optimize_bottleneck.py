"""Optimize a 2-token bottleneck to transmit answers through the vision channel.

Replaces the full vision pipeline with N learnable tokens and checks whether
a consistent codebook emerges (one code per answer choice).

All results are saved to results/two_bit/ (separate from the main optimization
results in results/optimization/).

Usage:
    # Default: MMMU, 2 tokens, 50 questions, 50 steps
    python scripts/optimize_bottleneck.py

    # Quick smoke test (1 question, 5 steps)
    python scripts/optimize_bottleneck.py --max_samples 1 --num_steps 5

    # More tokens or different benchmark
    python scripts/optimize_bottleneck.py --benchmark scienceqa --num_tokens 4

    # Single GPU (e.g. 3070/3090)
    python scripts/optimize_bottleneck.py --gpu_id 0

    # Skip codebook analysis (just optimise)
    python scripts/optimize_bottleneck.py --analyse false
"""

import os
import sys
import logging
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("optimize_bottleneck")


def main(
    benchmark: str = "mmmu",
    gpu_id: int = 0,
    max_samples: int = 50,
    num_tokens: int = 1,
    num_steps: int = 50,
    lr: float = 0.01,
    snapshot_every: int = 5,
    model_path: str = "nyu-visionx/cambrian-8b",
    conv_mode: str = "llama_3",
    results_dir: str = "results/two_bit",
    analyse: bool = True,
    top_k: int = 10,
):
    """Run 2-token bottleneck optimisation and codebook analysis.

    Args:
        benchmark: Benchmark name (mmmu, scienceqa, pope, etc.).
        gpu_id: Single GPU index.
        max_samples: Number of questions to optimise.
        num_tokens: Number of bottleneck tokens (default 2).
        num_steps: Gradient descent steps per question.
        lr: Adam learning rate.
        snapshot_every: Check accuracy + decode tokens every N steps.
        model_path: HuggingFace model path.
        conv_mode: Conversation template.
        results_dir: Where to save all results (default: results/two_bit/).
        analyse: Run codebook analysis after optimisation.
        top_k: Top-k tokens to decode through LM head.
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "optimize_bottleneck.log")),
        ],
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"GPU: {gpu_id}")
    logger.info(f"Bottleneck: {num_tokens} tokens x hidden_dim")
    logger.info(f"Optimisation: {num_steps} steps, lr={lr}, snapshots every {snapshot_every}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Results dir: {results_dir}")

    # Load model on single GPU
    from src.model.loader import load_cambrian

    logger.info("Loading model...")
    tokenizer, model, image_processor, context_len = load_cambrian(
        model_path=model_path,
        gpu_ids=[0],  # Remapped: CUDA_VISIBLE_DEVICES makes physical gpu_id -> logical 0
    )
    logger.info("Model loaded")

    # Load benchmark
    from src.benchmarks import get_benchmark

    bench = get_benchmark(benchmark)
    bench.load(max_samples=max_samples)
    logger.info(f"Loaded {len(bench)} samples from {benchmark}")

    # Run optimisation
    from src.optimization.bottleneck_optimizer import (
        optimize_bottleneck_per_question,
        analyse_codebook,
    )

    logger.info("=" * 60)
    logger.info("PHASE: Bottleneck per-question optimisation")
    logger.info("=" * 60)

    summary = optimize_bottleneck_per_question(
        model=model,
        tokenizer=tokenizer,
        benchmark=bench,
        image_processor=image_processor,
        num_tokens=num_tokens,
        max_samples=max_samples,
        num_steps=num_steps,
        lr=lr,
        conv_mode=conv_mode,
        results_dir=results_dir,
        snapshot_every=snapshot_every,
    )
    logger.info(f"Optimisation results: {summary}")

    # Codebook analysis
    if analyse:
        logger.info("=" * 60)
        logger.info("PHASE: Codebook analysis")
        logger.info("=" * 60)

        codebook = analyse_codebook(
            results_dir=results_dir,
            benchmark_name=benchmark,
            model=model,
            tokenizer=tokenizer,
            top_k=top_k,
        )

        # Print readable summary
        print("\n" + "=" * 60)
        print("CODEBOOK ANALYSIS")
        print("=" * 60)

        if codebook.get("answer_classes"):
            print("\nPer-answer statistics:")
            for answer, stats in sorted(codebook["answer_classes"].items()):
                print(
                    f"  {answer}: {stats['count']} samples, "
                    f"acc={stats['accuracy']:.1f}%, "
                    f"spread={stats['within_class_spread_mean']:.2f}"
                )

        if codebook.get("between_class_distances"):
            print("\nBetween-class distances:")
            for pair, dists in sorted(codebook["between_class_distances"].items()):
                print(
                    f"  {pair}: L2={dists['l2_distance']:.2f}, "
                    f"cos_sim={dists['cosine_similarity']:.4f}"
                )

        if codebook.get("confusion_matrix"):
            print("\nConfusion matrix (rows=ground truth, cols=predicted):")
            answers = sorted(codebook["confusion_matrix"].keys())
            header = "      " + "  ".join(f"{a:>4}" for a in answers)
            print(header)
            for gt in answers:
                row = codebook["confusion_matrix"][gt]
                vals = "  ".join(f"{row.get(p, 0):>4}" for p in answers)
                print(f"  {gt:>2}:  {vals}")

        if codebook.get("step_accuracy_curve"):
            curve = codebook["step_accuracy_curve"]
            print(f"\nStep-accuracy evolution:")
            for pt in curve:
                bar = "#" * int(pt["accuracy"] / 2)
                print(f"  step {pt['step']:>3}: {pt['accuracy']:>5.1f}% {bar}")

        if codebook.get("lm_head_decoding"):
            print("\nWhat the LLM reads from each answer's centroid tokens:")
            for answer, token_decodings in sorted(codebook["lm_head_decoding"].items()):
                decoded_str = " | ".join(
                    f"pos{i}: '{d[0]['token']}'({d[0]['prob']:.3f})"
                    for i, d in enumerate(token_decodings)
                )
                print(f"  {answer} -> [{decoded_str}]")

        if codebook.get("pca"):
            pca_info = codebook["pca"]
            print(
                f"\nPCA explained variance: {pca_info['explained_variance_ratio']}"
            )

        print(f"\nAll results saved to: {os.path.abspath(results_dir)}/")
        print(f"  {benchmark}_bottleneck.jsonl         — per-question results")
        print(f"  {benchmark}_bottleneck_summary.json  — summary stats")
        print(f"  {benchmark}_codebook_analysis.json   — codebook + graph data")
        print(f"  tensors/                             — saved token pairs")

    logger.info("All done!")


if __name__ == "__main__":
    fire.Fire(main)
