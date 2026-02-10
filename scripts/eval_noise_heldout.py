"""Evaluate optimized noise embeddings on held-out (unseen) questions.

Loads saved universal embeddings and tests them on questions the
optimization never saw, to measure generalization vs overfitting.

Also re-evaluates on the training questions for direct comparison.

Usage:
    python scripts/eval_noise_heldout.py
    python scripts/eval_noise_heldout.py --train_samples 50 --test_samples 100
    python scripts/eval_noise_heldout.py --checkpoint results/optimization/tensors/mmmu_universal_epoch25.pt
"""

import os
import sys
import json
import logging
import torch
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_noise_heldout")


def _check_answer(model, tokenizer, features, question_text, benchmark, sample, device, conv_mode):
    """Single forward pass + argmax to check if model answers correctly."""
    from cambrian.mm_utils import tokenizer_image_token
    from cambrian.constants import IMAGE_TOKEN_INDEX
    from src.model.inference import build_prompt
    from src.optimization.utils import encode_images_hook

    with torch.no_grad(), encode_images_hook(model, features):
        output = model.forward(
            input_ids=tokenizer_image_token(
                prompt=build_prompt(question_text, conv_mode=conv_mode, include_image=True),
                tokenizer=tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device),
            images=[
                torch.zeros(1, 3, 384, 384, device=device, dtype=model.dtype)
                for _ in range(4)
            ],
            image_sizes=[(384, 384)],
        )
        next_token_id = output.logits[0, -1].argmax().item()
        response = tokenizer.decode([next_token_id]).strip()

    prediction = benchmark.extract_answer(response, sample)
    correct = benchmark.score(prediction, sample)
    return response, prediction, correct


def main(
    benchmark: str = "mmmu",
    checkpoint: str = "results/optimization/tensors/mmmu_universal.pt",
    gpu_ids: str = "0,1,2,3",
    model_path: str = "nyu-visionx/cambrian-8b",
    conv_mode: str = "llama_3",
    train_samples: int = 50,
    test_samples: int = 200,
    output_dir: str = "results/optimization/heldout_eval",
):
    """Evaluate optimized embeddings on held-out questions.

    Args:
        benchmark: Benchmark name.
        checkpoint: Path to saved universal embedding tensors.
        gpu_ids: Comma-separated GPU indices.
        model_path: HuggingFace model path.
        conv_mode: Conversation template.
        train_samples: Number of samples used for training (to skip).
        test_samples: Number of held-out samples to evaluate on.
        output_dir: Where to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse GPUs
    if isinstance(gpu_ids, (list, tuple)):
        gpu_list = [int(x) for x in gpu_ids]
    else:
        gpu_list = [int(x) for x in str(gpu_ids).split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    remapped_gpus = list(range(len(gpu_list)))

    # Load model
    from src.model.loader import load_cambrian
    logger.info("Loading model...")
    tokenizer, model, image_processor, context_len = load_cambrian(
        model_path=model_path,
        gpu_ids=remapped_gpus,
    )
    device = next(model.parameters()).device
    logger.info("Model loaded")

    # Load benchmark — FULL dataset, not limited to train_samples
    from src.benchmarks import get_benchmark
    bench = get_benchmark(benchmark)
    bench.load()  # No max_samples — loads everything
    all_samples = list(bench)
    logger.info(f"Loaded {len(all_samples)} total samples from {benchmark}")

    # Split into train and test
    train_set = all_samples[:train_samples]
    test_set = all_samples[train_samples:train_samples + test_samples]
    logger.info(f"Train set: {len(train_set)} samples (used during optimization)")
    logger.info(f"Test set:  {len(test_set)} samples (NEVER seen during optimization)")

    # Verify no overlap
    train_ids = {s.question_id for s in train_set}
    test_ids = {s.question_id for s in test_set}
    overlap = train_ids & test_ids
    if overlap:
        logger.error(f"OVERLAP DETECTED: {len(overlap)} shared question IDs!")
        return
    logger.info(f"Verified: zero overlap between train and test sets")

    # Load optimized embeddings
    logger.info(f"Loading embeddings from {checkpoint}")
    features = torch.load(checkpoint, map_location=device, weights_only=True)
    features = [t.to(device=device, dtype=torch.float32) for t in features]

    # Offload vision encoders (same as during training)
    inner = getattr(model, "model", model)
    towers = getattr(inner, "vision_tower_aux_list", None)
    if towers:
        for tower in towers:
            tower.cpu()
        torch.cuda.empty_cache()
        logger.info(f"Moved {len(towers)} vision encoders to CPU")

    # Enable gradient checkpointing for memory (eval still benefits)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # --- Evaluate on TRAIN set (expect high accuracy — these were optimized on) ---
    logger.info("=" * 60)
    logger.info("Evaluating on TRAIN set (sanity check)...")
    logger.info("=" * 60)
    train_correct = 0
    train_results = []

    for i, sample in enumerate(train_set):
        question_text = bench.format_question(sample)
        response, prediction, correct = _check_answer(
            model, tokenizer, features, question_text, bench, sample, device, conv_mode,
        )
        if correct:
            train_correct += 1
        train_results.append({
            "question_id": sample.question_id,
            "prediction": prediction,
            "ground_truth": sample.ground_truth,
            "correct": correct,
            "split": "train",
        })
        if (i + 1) % 10 == 0:
            acc = train_correct / (i + 1) * 100
            logger.info(f"  Train [{i + 1}/{len(train_set)}]: acc={acc:.1f}%")

    train_acc = train_correct / len(train_set) * 100
    logger.info(f"TRAIN accuracy: {train_correct}/{len(train_set)} = {train_acc:.1f}%")

    # --- Evaluate on TEST set (the real test — never seen during optimization) ---
    logger.info("=" * 60)
    logger.info("Evaluating on HELD-OUT TEST set...")
    logger.info("=" * 60)
    test_correct = 0
    test_results = []

    for i, sample in enumerate(test_set):
        question_text = bench.format_question(sample)
        response, prediction, correct = _check_answer(
            model, tokenizer, features, question_text, bench, sample, device, conv_mode,
        )
        if correct:
            test_correct += 1
        test_results.append({
            "question_id": sample.question_id,
            "prediction": prediction,
            "ground_truth": sample.ground_truth,
            "correct": correct,
            "split": "test",
        })
        if (i + 1) % 10 == 0:
            acc = test_correct / (i + 1) * 100
            logger.info(f"  Test [{i + 1}/{len(test_set)}]: acc={acc:.1f}%")

    test_acc = test_correct / len(test_set) * 100
    logger.info(f"TEST accuracy: {test_correct}/{len(test_set)} = {test_acc:.1f}%")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("HELD-OUT EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Benchmark:  {benchmark}")
    print()
    print(f"  TRAIN accuracy ({len(train_set)} questions):  {train_acc:.1f}%  (optimized on these)")
    print(f"  TEST accuracy  ({len(test_set)} questions):  {test_acc:.1f}%  (NEVER seen)")
    print()
    print(f"  Overfitting gap: {train_acc - test_acc:.1f} pp")
    print()
    print(f"  Reference — MMMU normal (real images): ~38.7%")
    print(f"  Reference — MMMU no image:             ~31.1%")
    print(f"  Reference — Random chance (4-choice):   25.0%")
    print()

    if test_acc > 38.7:
        print("  >> TEST accuracy BEATS real images — noise generalizes!")
    elif test_acc > 31.1:
        print("  >> TEST accuracy beats no-image but below real images")
    elif test_acc > 25.0:
        print("  >> TEST accuracy above chance but below no-image baseline")
    else:
        print("  >> TEST accuracy at or below chance — pure overfitting")

    # Save results
    summary = {
        "benchmark": benchmark,
        "checkpoint": checkpoint,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "train_accuracy": round(train_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "overfitting_gap": round(train_acc - test_acc, 2),
        "train_question_ids": [s.question_id for s in train_set],
        "test_question_ids": [s.question_id for s in test_set],
    }

    summary_path = os.path.join(output_dir, f"{benchmark}_heldout_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    results_path = os.path.join(output_dir, f"{benchmark}_heldout_results.jsonl")
    with open(results_path, "w") as f:
        for r in train_results + test_results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
