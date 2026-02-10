"""Embedding-space noise optimization for Cambrian-8B.

Optimizes 4 learnable tensors (one per vision encoder output) to maximize
the probability of the correct answer. Bypasses vision encoders entirely.
"""

import json
import os
import time
import logging
import torch

from src.benchmarks.base import Benchmark, BenchmarkSample
from src.optimization.utils import encode_images_hook, get_encoder_output_shapes
from src.optimization.teacher_forcing import compute_teacher_forcing_loss

logger = logging.getLogger(__name__)


def _init_features(
    shapes: list[tuple[int, ...]],
    device: torch.device,
    dtype: torch.dtype,
    init_scale: float = 0.02,
) -> list[torch.Tensor]:
    """Initialize random learnable feature tensors."""
    features = []
    for shape in shapes:
        t = torch.randn(shape, device=device, dtype=dtype) * init_scale
        t.requires_grad_(True)
        features.append(t)
    return features


def optimize_per_question(
    model,
    tokenizer,
    image_processor,
    benchmark: Benchmark,
    max_samples: int = 50,
    num_steps: int = 50,
    lr: float = 0.01,
    conv_mode: str = "llama_3",
    results_dir: str = "results/optimization",
) -> dict:
    """Optimize embedding-space noise independently for each question.

    For each question:
      1. Initialize 4 random feature tensors
      2. Run num_steps of gradient descent to minimize teacher-forcing loss
      3. After optimization, check if the model now answers correctly
      4. Record results (written to JSONL in real-time)

    Args:
        model: Loaded Cambrian model.
        tokenizer: Tokenizer.
        image_processor: Image processor (used to discover encoder output shapes).
        benchmark: Loaded benchmark instance.
        max_samples: Number of questions to optimize.
        num_steps: Gradient descent steps per question.
        lr: Adam learning rate.
        conv_mode: Conversation template.
        results_dir: Where to save results.

    Returns:
        Summary dict with accuracy metrics.
    """
    os.makedirs(results_dir, exist_ok=True)
    device = next(model.parameters()).device
    dtype = model.dtype

    # Discover encoder output shapes from a dummy forward pass
    shapes = get_encoder_output_shapes(model, image_processor)

    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    results = []
    correct_after = 0

    samples = list(benchmark)
    if max_samples:
        samples = samples[:max_samples]

    # JSONL file for real-time results (append mode)
    results_file = os.path.join(results_dir, f"{benchmark.name}_optimized_embeddings.jsonl")

    # Check for existing results (checkpoint/resume)
    completed_ids = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        completed_ids.add(r["question_id"])
                        results.append(r)
                        if r.get("correct"):
                            correct_after += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} questions already done")

    total = len(samples)
    run_start = time.time()

    logger.info(f"Starting optimization: {total} questions, {num_steps} steps each, lr={lr}")
    logger.info(f"Results file: {results_file}")

    for sample_idx, sample in enumerate(samples):
        if sample.question_id in completed_ids:
            continue

        question_text = benchmark.format_question(sample)
        answer_text = sample.ground_truth

        # Initialize fresh features for this question
        features = _init_features(shapes, device, dtype)
        optimizer = torch.optim.Adam(features, lr=lr)

        # Measure initial loss (random embeddings)
        with torch.no_grad(), encode_images_hook(model, features):
            initial_loss = compute_teacher_forcing_loss(
                model, tokenizer, question_text, answer_text,
                conv_mode=conv_mode,
            ).item()

        # Optimization loop with per-step logging
        losses = []
        start_time = time.time()

        for step in range(num_steps):
            optimizer.zero_grad()

            with encode_images_hook(model, features):
                loss = compute_teacher_forcing_loss(
                    model, tokenizer, question_text, answer_text,
                    conv_mode=conv_mode,
                )

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Log every 10 steps within a question
            if (step + 1) % 10 == 0:
                logger.info(
                    f"  Q{sample_idx + 1}/{total} step {step + 1}/{num_steps}: "
                    f"loss {losses[-1]:.4f} (started at {initial_loss:.4f})"
                )

        opt_time = time.time() - start_time

        # Check accuracy AFTER optimization: generate an answer
        with torch.no_grad(), encode_images_hook(model, features):
            from src.model.inference import run_inference
            from PIL import Image
            dummy_image = Image.new("RGB", (384, 384), color=(128, 128, 128))
            response = run_inference(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                question=question_text,
                image=dummy_image,
                conv_mode=conv_mode,
                max_new_tokens=32,
            )

        prediction = benchmark.extract_answer(response, sample)
        correct = benchmark.score(prediction, sample)
        if correct:
            correct_after += 1

        result = {
            "question_id": sample.question_id,
            "question": sample.question[:200],
            "ground_truth": answer_text,
            "prediction": prediction,
            "raw_response": response[:200],
            "correct": correct,
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(losses[-1], 4),
            "loss_reduction": round(initial_loss - losses[-1], 4),
            "optimization_time_s": round(opt_time, 1),
            "num_steps": num_steps,
        }
        results.append(result)

        # Write result immediately (real-time progress)
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Per-question summary log
        done = len(results)
        acc = correct_after / done * 100
        elapsed = time.time() - run_start
        remaining = total - done
        avg_time = elapsed / (done - len(completed_ids)) if done > len(completed_ids) else 0
        eta = remaining * avg_time

        status = "CORRECT" if correct else "WRONG"
        logger.info(
            f"[{done}/{total}] {sample.question_id}: {status} "
            f"(pred={prediction}, gt={answer_text}) "
            f"loss {initial_loss:.3f}->{losses[-1]:.3f} "
            f"| acc={acc:.1f}% | {opt_time:.1f}s | ETA {eta / 60:.0f}min"
        )

    # Disable gradient checkpointing after optimization
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # Summary
    accuracy = correct_after / len(results) * 100 if results else 0
    avg_loss_reduction = sum(r["loss_reduction"] for r in results) / len(results) if results else 0

    summary = {
        "benchmark": benchmark.name,
        "num_samples": len(results),
        "num_steps": num_steps,
        "learning_rate": lr,
        "accuracy_after_optimization": round(accuracy, 2),
        "avg_initial_loss": round(sum(r["initial_loss"] for r in results) / len(results), 4),
        "avg_final_loss": round(sum(r["final_loss"] for r in results) / len(results), 4),
        "avg_loss_reduction": round(avg_loss_reduction, 4),
    }

    summary_file = os.path.join(results_dir, f"{benchmark.name}_optimized_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Optimization complete: {summary}")
    return summary
