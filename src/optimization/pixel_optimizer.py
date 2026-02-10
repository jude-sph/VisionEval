"""Pixel-space noise optimization for Cambrian-8B.

Optimizes raw pixel values [1, 3, 384, 384] that, when fed through the real
vision encoders, maximize P(correct answer). Produces visible noise images.

Two modes:
  - Universal: ONE noise image optimized across all questions
  - Per-question: fresh noise image optimized independently per question
"""

import json
import os
import time
import logging
import torch
import numpy as np
from PIL import Image as PILImage

from cambrian.mm_utils import tokenizer_image_token
from cambrian.constants import IMAGE_TOKEN_INDEX

from src.benchmarks.base import Benchmark, BenchmarkSample
from src.model.inference import build_prompt
from src.optimization.utils import enable_vision_grad
from src.optimization.teacher_forcing import compute_teacher_forcing_loss
from src.optimization.differentiable_preprocess import (
    extract_preprocess_params,
    differentiable_preprocess,
)

logger = logging.getLogger(__name__)

# ConvNeXt (encoder index 3) takes 1024x1024 input and produces 9216 tokens.
# Its backward pass stores massive intermediate activations that OOM on 12GB GPUs.
# We detach its preprocessed input so it still contributes to the forward pass
# but gradients only flow through the other 3 encoders (SigLIP, CLIP, DINOv2).
CONVNEXT_INDEX = 3


def _init_pixels(
    device: torch.device,
    size: tuple[int, int] = (384, 384),
) -> torch.Tensor:
    """Initialize a random pixel tensor in [0, 1].

    Args:
        device: Device to place tensor on.
        size: (H, W) spatial dimensions.

    Returns:
        Float32 tensor [1, 3, H, W] with requires_grad=True.
    """
    pixels = torch.rand(1, 3, size[0], size[1], device=device, dtype=torch.float32)
    pixels.requires_grad_(True)
    return pixels


def _save_pixel_image(pixels: torch.Tensor, path: str):
    """Save a pixel tensor as a PNG image.

    Args:
        pixels: Tensor [1, 3, H, W] in [0, 1] range, any dtype.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_np = pixels.detach().cpu().clamp(0, 1).squeeze(0)  # [3, H, W]
    img_np = (img_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [H, W, 3]
    PILImage.fromarray(img_np).save(path)


def _setup_model_for_pixel_optimization(model, image_processor):
    """Setup model for pixel optimization: keep encoders on GPU, enable grad.

    Unlike embedding optimization, we do NOT offload vision encoders to CPU
    because we need them for the forward pass through real encoders.

    Returns:
        (preprocess_params, device)
    """
    device = next(model.parameters()).device

    num_devices = len(set(p.device for p in model.parameters()))
    logger.info(f"Model spread across {num_devices} device(s)")

    # Extract preprocessing params from each encoder's image processor
    preprocess_params = extract_preprocess_params(image_processor, device)

    # Enable gradient checkpointing (train mode required)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled (non-reentrant, train mode)")

    return preprocess_params, device


def _check_answer_pixel(
    model, tokenizer, pixels, preprocess_params, question_text,
    benchmark, sample, device, conv_mode,
):
    """Forward pass through real encoders + argmax to check answer."""
    preprocessed = differentiable_preprocess(pixels, preprocess_params)

    with torch.no_grad():
        output = model.forward(
            input_ids=tokenizer_image_token(
                prompt=build_prompt(question_text, conv_mode=conv_mode, include_image=True),
                tokenizer=tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device),
            images=preprocessed,
            image_sizes=[(384, 384)],
        )
        next_token_id = output.logits[0, -1].argmax().item()
        response = tokenizer.decode([next_token_id]).strip()

    prediction = benchmark.extract_answer(response, sample)
    correct = benchmark.score(prediction, sample)
    return response, prediction, correct


def optimize_pixel_universal(
    model,
    tokenizer,
    image_processor,
    benchmark: Benchmark,
    max_samples: int = 50,
    num_epochs: int = 10,
    lr: float = 0.01,
    conv_mode: str = "llama_3",
    results_dir: str = "results/optimization",
) -> dict:
    """Optimize ONE noise image across all questions.

    Trains a single pixel tensor by accumulating gradients across all
    questions per epoch, then taking one optimizer step + clamping to [0,1].

    Args:
        model: Loaded Cambrian model.
        tokenizer: Tokenizer.
        image_processor: Image processor (list of 4 per-encoder processors).
        benchmark: Loaded benchmark instance.
        max_samples: Number of questions to train on.
        num_epochs: Number of full passes through all questions.
        lr: Adam learning rate.
        conv_mode: Conversation template.
        results_dir: Where to save results.

    Returns:
        Summary dict with accuracy metrics.
    """
    pixel_dir = os.path.join(results_dir, "pixel")
    os.makedirs(pixel_dir, exist_ok=True)

    preprocess_params, device = _setup_model_for_pixel_optimization(model, image_processor)

    samples = list(benchmark)
    if max_samples:
        samples = samples[:max_samples]
    total = len(samples)

    # Initialize random pixel image
    pixels = _init_pixels(device)
    optimizer = torch.optim.Adam([pixels], lr=lr)

    # Save initial random image
    _save_pixel_image(pixels, os.path.join(pixel_dir, f"{benchmark.name}_pixel_universal_initial.png"))
    torch.save(
        pixels.detach().cpu(),
        os.path.join(pixel_dir, f"{benchmark.name}_pixel_universal_initial.pt"),
    )

    epoch_log_file = os.path.join(pixel_dir, f"{benchmark.name}_pixel_universal_epochs.jsonl")
    run_start = time.time()

    logger.info(f"Starting PIXEL UNIVERSAL optimization: {total} questions, {num_epochs} epochs, lr={lr}")
    logger.info(f"  ConvNeXt (encoder {CONVNEXT_INDEX}) excluded from backward pass (OOM prevention)")

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        epoch_loss = 0
        per_question_losses = []
        nan_detected = False

        for sample_idx, sample in enumerate(samples):
            question_text = benchmark.format_question(sample)
            answer_text = sample.ground_truth

            # Differentiable preprocess → real encoders → LLM → loss
            preprocessed = differentiable_preprocess(pixels, preprocess_params)
            # Detach ConvNeXt input to avoid OOM on backward (still used in forward)
            preprocessed[CONVNEXT_INDEX] = preprocessed[CONVNEXT_INDEX].detach()

            with enable_vision_grad(model, exclude_indices=[CONVNEXT_INDEX]):
                loss = compute_teacher_forcing_loss(
                    model, tokenizer, question_text, answer_text,
                    conv_mode=conv_mode,
                    images=preprocessed,
                )

            # Average gradient across questions
            (loss / total).backward()
            loss_val = loss.item()

            if not torch.isfinite(torch.tensor(loss_val)):
                logger.warning(f"  Epoch {epoch + 1} Q{sample_idx + 1}: NaN/Inf loss")
                nan_detected = True
                break

            epoch_loss += loss_val
            per_question_losses.append(round(loss_val, 4))

        if nan_detected:
            logger.warning(f"  Epoch {epoch + 1}: NaN detected, stopping")
            break

        # Check gradient is flowing to pixels
        if epoch == 0 and pixels.grad is not None:
            grad_norm = pixels.grad.norm().item()
            logger.info(f"  Pixel gradient norm: {grad_norm:.6f}")
            if grad_norm == 0:
                logger.warning("  WARNING: Pixel gradient is zero! Gradients not flowing.")

        torch.nn.utils.clip_grad_norm_([pixels], max_norm=1.0)
        optimizer.step()

        # Clamp pixels to valid [0, 1] range
        pixels.data.clamp_(0, 1)

        avg_loss = epoch_loss / total
        epoch_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        # Check accuracy after each epoch
        epoch_correct = 0
        for sample in samples:
            question_text = benchmark.format_question(sample)
            _, _, correct = _check_answer_pixel(
                model, tokenizer, pixels, preprocess_params,
                question_text, benchmark, sample, device, conv_mode,
            )
            if correct:
                epoch_correct += 1
        epoch_acc = epoch_correct / total * 100
        epoch_accuracies.append(epoch_acc)

        elapsed = time.time() - run_start
        avg_epoch_time = elapsed / (epoch + 1)
        eta = (num_epochs - epoch - 1) * avg_epoch_time

        logger.info(
            f"  Epoch {epoch + 1}/{num_epochs}: avg_loss={avg_loss:.4f} acc={epoch_acc:.1f}% "
            f"| {epoch_time:.0f}s | ETA {eta / 60:.0f}min"
        )

        # Write epoch data in real-time
        epoch_data = {
            "epoch": epoch + 1,
            "avg_loss": round(avg_loss, 4),
            "accuracy": round(epoch_acc, 2),
            "per_question_losses": per_question_losses,
            "epoch_time_s": round(epoch_time, 1),
        }
        with open(epoch_log_file, "a") as f:
            f.write(json.dumps(epoch_data) + "\n")

        # Save image checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            _save_pixel_image(
                pixels,
                os.path.join(pixel_dir, f"{benchmark.name}_pixel_universal_epoch{epoch + 1}.png"),
            )
            torch.save(
                pixels.detach().cpu(),
                os.path.join(pixel_dir, f"{benchmark.name}_pixel_universal_epoch{epoch + 1}.pt"),
            )

    opt_time = time.time() - run_start
    logger.info(f"Pixel universal optimization done in {opt_time:.0f}s. Final evaluation...")

    # Save final image
    _save_pixel_image(pixels, os.path.join(pixel_dir, f"{benchmark.name}_pixel_universal.png"))
    torch.save(
        pixels.detach().cpu(),
        os.path.join(pixel_dir, f"{benchmark.name}_pixel_universal.pt"),
    )

    # Final per-question evaluation
    correct_count = 0
    per_question_results = []

    for sample_idx, sample in enumerate(samples):
        question_text = benchmark.format_question(sample)

        with torch.no_grad():
            preprocessed = differentiable_preprocess(pixels, preprocess_params)
            final_loss = compute_teacher_forcing_loss(
                model, tokenizer, question_text, sample.ground_truth,
                conv_mode=conv_mode,
                images=preprocessed,
            ).item()

        response, prediction, correct = _check_answer_pixel(
            model, tokenizer, pixels, preprocess_params,
            question_text, benchmark, sample, device, conv_mode,
        )
        if correct:
            correct_count += 1

        per_question_results.append({
            "question_id": sample.question_id,
            "question": sample.question[:200],
            "ground_truth": sample.ground_truth,
            "prediction": prediction,
            "raw_response": response[:200],
            "correct": correct,
            "final_loss": round(final_loss, 4),
        })

    accuracy = correct_count / total * 100 if total > 0 else 0

    # Write per-question eval results
    results_file = os.path.join(pixel_dir, f"{benchmark.name}_pixel_universal_results.jsonl")
    with open(results_file, "w") as f:
        for r in per_question_results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "mode": "pixel_universal",
        "benchmark": benchmark.name,
        "num_samples": total,
        "num_epochs": len(epoch_losses),
        "learning_rate": lr,
        "accuracy": round(accuracy, 2),
        "initial_avg_loss": round(epoch_losses[0], 4) if epoch_losses else 0,
        "final_avg_loss": round(epoch_losses[-1], 4) if epoch_losses else 0,
        "loss_curve": [round(l, 4) for l in epoch_losses],
        "accuracy_curve": [round(a, 2) for a in epoch_accuracies],
        "optimization_time_s": round(opt_time, 1),
    }

    summary_file = os.path.join(pixel_dir, f"{benchmark.name}_pixel_universal_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Pixel universal optimization complete: {summary}")
    return summary


def optimize_pixel_per_question(
    model,
    tokenizer,
    image_processor,
    benchmark: Benchmark,
    max_samples: int = 50,
    num_steps: int = 20,
    lr: float = 0.01,
    conv_mode: str = "llama_3",
    results_dir: str = "results/optimization",
    _skip_setup: bool = False,
    _preprocess_params=None,
) -> dict:
    """Optimize a noise image independently for each question.

    For each question:
      1. Initialize random pixel tensor [1, 3, 384, 384]
      2. Run num_steps of gradient descent through real vision encoders
      3. Save the optimized noise image as PNG
      4. Check if the model now answers correctly

    Args:
        model: Loaded Cambrian model.
        tokenizer: Tokenizer.
        image_processor: Image processor.
        benchmark: Loaded benchmark instance.
        max_samples: Number of questions to optimize.
        num_steps: Gradient descent steps per question.
        lr: Adam learning rate.
        conv_mode: Conversation template.
        results_dir: Where to save results.
        _skip_setup: If True, skip model setup (already done by universal).
        _preprocess_params: Pre-extracted params (when _skip_setup=True).

    Returns:
        Summary dict with accuracy metrics.
    """
    pixel_dir = os.path.join(results_dir, "pixel")
    images_dir = os.path.join(pixel_dir, "images")
    tensors_dir = os.path.join(pixel_dir, "tensors")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    if _skip_setup and _preprocess_params is not None:
        preprocess_params = _preprocess_params
        device = next(model.parameters()).device
    else:
        preprocess_params, device = _setup_model_for_pixel_optimization(model, image_processor)

    results = []
    correct_after = 0

    samples = list(benchmark)
    if max_samples:
        samples = samples[:max_samples]

    # JSONL file for real-time results (append mode)
    results_file = os.path.join(pixel_dir, f"{benchmark.name}_pixel_optimized.jsonl")

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

    logger.info(f"Starting PIXEL PER-QUESTION optimization: {total} questions, {num_steps} steps each, lr={lr}")
    logger.info(f"  ConvNeXt (encoder {CONVNEXT_INDEX}) excluded from backward pass (OOM prevention)")
    logger.info(f"Results file: {results_file}")

    for sample_idx, sample in enumerate(samples):
        if sample.question_id in completed_ids:
            continue

        question_text = benchmark.format_question(sample)
        answer_text = sample.ground_truth

        # Initialize fresh pixel tensor for this question
        pixels = _init_pixels(device)
        optimizer = torch.optim.Adam([pixels], lr=lr)

        # Measure initial loss and answer BEFORE optimization
        preprocessed = differentiable_preprocess(pixels, preprocess_params)
        with torch.no_grad():
            initial_loss = compute_teacher_forcing_loss(
                model, tokenizer, question_text, answer_text,
                conv_mode=conv_mode,
                images=preprocessed,
            ).item()
        initial_response, initial_prediction, initial_correct = _check_answer_pixel(
            model, tokenizer, pixels, preprocess_params,
            question_text, benchmark, sample, device, conv_mode,
        )

        # Optimization loop
        losses = []
        start_time = time.time()
        nan_detected = False

        for step in range(num_steps):
            optimizer.zero_grad()

            preprocessed = differentiable_preprocess(pixels, preprocess_params)
            # Detach ConvNeXt input to avoid OOM on backward (still used in forward)
            preprocessed[CONVNEXT_INDEX] = preprocessed[CONVNEXT_INDEX].detach()

            with enable_vision_grad(model, exclude_indices=[CONVNEXT_INDEX]):
                loss = compute_teacher_forcing_loss(
                    model, tokenizer, question_text, answer_text,
                    conv_mode=conv_mode,
                    images=preprocessed,
                )

            loss.backward()

            # Check gradient on first step of first question
            if sample_idx == 0 and step == 0 and pixels.grad is not None:
                grad_norm = pixels.grad.norm().item()
                logger.info(f"  Pixel gradient norm (step 0): {grad_norm:.6f}")

            torch.nn.utils.clip_grad_norm_([pixels], max_norm=1.0)
            optimizer.step()

            # Clamp to valid pixel range
            pixels.data.clamp_(0, 1)

            loss_val = loss.item()

            if not torch.isfinite(torch.tensor(loss_val)):
                logger.warning(
                    f"  Q{sample_idx + 1}/{total} step {step + 1}: NaN/Inf loss, stopping early"
                )
                nan_detected = True
                break

            losses.append(loss_val)

            if (step + 1) % 10 == 0:
                logger.info(
                    f"  Q{sample_idx + 1}/{total} step {step + 1}/{num_steps}: "
                    f"loss {losses[-1]:.4f} (started at {initial_loss:.4f})"
                )

        opt_time = time.time() - start_time
        final_loss = losses[-1] if losses else float("nan")

        # Save optimized noise image
        safe_id = str(sample.question_id).replace("/", "_")
        _save_pixel_image(pixels, os.path.join(images_dir, f"{safe_id}.png"))
        torch.save(
            pixels.detach().cpu(),
            os.path.join(tensors_dir, f"{safe_id}.pt"),
        )

        # Check accuracy AFTER optimization
        response, prediction, correct = _check_answer_pixel(
            model, tokenizer, pixels, preprocess_params,
            question_text, benchmark, sample, device, conv_mode,
        )
        if correct:
            correct_after += 1

        result = {
            "question_id": sample.question_id,
            "question": sample.question[:200],
            "ground_truth": answer_text,
            # Before optimization
            "initial_prediction": initial_prediction,
            "initial_response": initial_response[:200],
            "initial_correct": initial_correct,
            "initial_loss": round(initial_loss, 4),
            # After optimization
            "prediction": prediction,
            "raw_response": response[:200],
            "correct": correct,
            "final_loss": round(final_loss, 4) if not nan_detected else None,
            "loss_reduction": round(initial_loss - final_loss, 4) if not nan_detected else None,
            "loss_curve": [round(l, 4) for l in losses],
            "optimization_time_s": round(opt_time, 1),
            "num_steps": len(losses),
            "nan_detected": nan_detected,
        }
        results.append(result)

        # Write result immediately
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
        loss_str = f"{initial_loss:.3f}->{final_loss:.3f}" if not nan_detected else f"{initial_loss:.3f}->NaN"
        logger.info(
            f"[{done}/{total}] {sample.question_id}: {status} "
            f"(pred={prediction}, gt={answer_text}) "
            f"loss {loss_str} "
            f"| acc={acc:.1f}% | {opt_time:.1f}s | ETA {eta / 60:.0f}min"
        )

    # Disable gradient checkpointing after optimization
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # Summary
    accuracy = correct_after / len(results) * 100 if results else 0
    valid_results = [r for r in results if r.get("loss_reduction") is not None]
    avg_loss_reduction = sum(r["loss_reduction"] for r in valid_results) / len(valid_results) if valid_results else 0
    nan_count = sum(1 for r in results if r.get("nan_detected"))

    summary = {
        "mode": "pixel_per_question",
        "benchmark": benchmark.name,
        "num_samples": len(results),
        "num_nan": nan_count,
        "num_steps": num_steps,
        "learning_rate": lr,
        "accuracy_after_optimization": round(accuracy, 2),
        "avg_initial_loss": round(sum(r["initial_loss"] for r in results) / len(results), 4) if results else 0,
        "avg_final_loss": round(sum(r["final_loss"] for r in valid_results) / len(valid_results), 4) if valid_results else 0,
        "avg_loss_reduction": round(avg_loss_reduction, 4),
    }

    summary_file = os.path.join(pixel_dir, f"{benchmark.name}_pixel_optimized_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Pixel per-question optimization complete: {summary}")
    return summary
