"""Embedding-space noise optimization for Cambrian-8B.

Optimizes learnable tensors (one per vision encoder output) to maximize
the probability of the correct answer. Bypasses vision encoders entirely.

Two modes:
  - Universal: ONE set of embeddings optimized across all questions
  - Per-question: fresh embeddings optimized independently per question
"""

import json
import os
import time
import logging
import torch

from cambrian.mm_utils import tokenizer_image_token
from cambrian.constants import IMAGE_TOKEN_INDEX

from src.benchmarks.base import Benchmark, BenchmarkSample
from src.model.inference import build_prompt
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


def _setup_model_for_optimization(model, image_processor):
    """Shared setup: discover shapes, offload encoders, enable checkpointing.

    Returns:
        (shapes, device, dtype)
    """
    device = next(model.parameters()).device
    dtype = torch.float32

    num_devices = len(set(p.device for p in model.parameters()))
    logger.info(f"Model spread across {num_devices} device(s)")

    shapes = get_encoder_output_shapes(model, image_processor)

    # Free vision encoder memory — encode_images is patched so they're unused.
    inner = getattr(model, "model", model)
    towers = getattr(inner, "vision_tower_aux_list", None)
    if towers:
        for tower in towers:
            tower.cpu()
        torch.cuda.empty_cache()
        logger.info(f"Moved {len(towers)} vision encoders to CPU")

    # Enable gradient checkpointing (train mode required).
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled (non-reentrant, train mode)")

    return shapes, device, dtype


def _check_answer(model, tokenizer, features, question_text, benchmark, sample, device, conv_mode):
    """Single forward pass + argmax to check if model answers correctly."""
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


def optimize_universal(
    model,
    tokenizer,
    image_processor,
    benchmark: Benchmark,
    max_samples: int = 50,
    num_epochs: int = 10,
    lr: float = 0.001,
    conv_mode: str = "llama_3",
    results_dir: str = "results/optimization",
    resume: bool = False,
) -> dict:
    """Optimize ONE set of embeddings across all questions.

    Trains shared feature tensors by accumulating gradients across all
    questions per epoch, then taking one optimizer step.

    Args:
        model: Loaded Cambrian model.
        tokenizer: Tokenizer.
        image_processor: Image processor (used to discover encoder output shapes).
        benchmark: Loaded benchmark instance.
        max_samples: Number of questions to train on.
        num_epochs: Total number of epochs (including already-completed ones).
        lr: Adam learning rate.
        conv_mode: Conversation template.
        results_dir: Where to save results.
        resume: If True, load saved tensors and epoch log, continue training.

    Returns:
        Summary dict with accuracy metrics.
    """
    os.makedirs(results_dir, exist_ok=True)
    shapes, device, dtype = _setup_model_for_optimization(model, image_processor)

    samples = list(benchmark)
    if max_samples:
        samples = samples[:max_samples]
    total = len(samples)

    tensors_dir = os.path.join(results_dir, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)

    epoch_log_file = os.path.join(results_dir, f"{benchmark.name}_universal_epochs.jsonl")
    start_epoch = 0
    epoch_losses = []
    epoch_accuracies = []

    if resume:
        # Find the latest checkpoint tensor
        best_checkpoint = None
        best_epoch = 0
        for f_name in sorted(os.listdir(tensors_dir)):
            if f_name.startswith(f"{benchmark.name}_universal_epoch") and f_name.endswith(".pt"):
                ep = int(f_name.replace(f"{benchmark.name}_universal_epoch", "").replace(".pt", ""))
                if ep > best_epoch:
                    best_epoch = ep
                    best_checkpoint = os.path.join(tensors_dir, f_name)

        if best_checkpoint and best_epoch > 0:
            saved_tensors = torch.load(best_checkpoint, map_location=device, weights_only=True)
            features = []
            for t in saved_tensors:
                t = t.to(device=device, dtype=dtype)
                t.requires_grad_(True)
                features.append(t)
            start_epoch = best_epoch
            logger.info(f"Resumed from {best_checkpoint} (epoch {best_epoch})")

            # Reload epoch log to restore loss/accuracy curves
            if os.path.exists(epoch_log_file):
                with open(epoch_log_file) as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                if entry["epoch"] <= best_epoch:
                                    epoch_losses.append(entry["avg_loss"])
                                    epoch_accuracies.append(entry["accuracy"])
                            except (json.JSONDecodeError, KeyError):
                                pass
                logger.info(f"Restored {len(epoch_losses)} epoch log entries")
        else:
            logger.warning("No checkpoint found, starting from scratch")
            resume = False

    if not resume:
        # One set of features for ALL questions
        features = _init_features(shapes, device, dtype)

        # Save initial (random) embeddings for comparison
        torch.save(
            [f.detach().cpu() for f in features],
            os.path.join(tensors_dir, f"{benchmark.name}_universal_initial.pt"),
        )

    optimizer = torch.optim.Adam(features, lr=lr)

    # Restore optimizer state if resuming (preserves momentum/variance estimates)
    if resume and best_epoch > 0:
        opt_state_path = os.path.join(tensors_dir, f"{benchmark.name}_universal_optimizer.pt")
        if os.path.exists(opt_state_path):
            optimizer.load_state_dict(torch.load(opt_state_path, map_location=device, weights_only=True))
            logger.info(f"Restored optimizer state from {opt_state_path}")
        else:
            logger.warning("No optimizer state found, using fresh Adam (may cause brief loss spike)")

    results_file = os.path.join(results_dir, f"{benchmark.name}_universal_embeddings.jsonl")
    run_start = time.time()

    if start_epoch >= num_epochs:
        logger.info(f"Already completed {start_epoch} epochs (requested {num_epochs}), skipping training")
    else:
        logger.info(
            f"Starting UNIVERSAL optimization: {total} questions, "
            f"epochs {start_epoch + 1}-{num_epochs}, lr={lr}"
        )
    logger.info(f"Results file: {results_file}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        epoch_loss = 0
        per_question_losses = []
        nan_detected = False

        for sample_idx, sample in enumerate(samples):
            question_text = benchmark.format_question(sample)
            answer_text = sample.ground_truth

            with encode_images_hook(model, features):
                loss = compute_teacher_forcing_loss(
                    model, tokenizer, question_text, answer_text,
                    conv_mode=conv_mode,
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

        torch.nn.utils.clip_grad_norm_(features, max_norm=1.0)
        optimizer.step()

        avg_loss = epoch_loss / total
        epoch_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        # Check accuracy after each epoch (adds ~2s × N questions but very informative)
        epoch_correct = 0
        for sample in samples:
            question_text = benchmark.format_question(sample)
            _, _, correct = _check_answer(
                model, tokenizer, features, question_text, benchmark, sample, device, conv_mode,
            )
            if correct:
                epoch_correct += 1
        epoch_acc = epoch_correct / total * 100
        epoch_accuracies.append(epoch_acc)

        elapsed = time.time() - run_start
        # Use elapsed / completed epochs for more accurate ETA
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

        # Save embedding checkpoint + optimizer state every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save(
                [f.detach().cpu() for f in features],
                os.path.join(tensors_dir, f"{benchmark.name}_universal_epoch{epoch + 1}.pt"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(tensors_dir, f"{benchmark.name}_universal_optimizer.pt"),
            )

    opt_time = time.time() - run_start
    logger.info(f"Universal optimization done in {opt_time:.0f}s. Final evaluation...")

    # Final evaluation: detailed per-question results with the optimized embeddings
    correct_count = 0
    per_question_results = []

    for sample_idx, sample in enumerate(samples):
        question_text = benchmark.format_question(sample)

        # Get per-question loss with final embeddings
        with torch.no_grad(), encode_images_hook(model, features):
            final_loss = compute_teacher_forcing_loss(
                model, tokenizer, question_text, sample.ground_truth,
                conv_mode=conv_mode,
            ).item()

        response, prediction, correct = _check_answer(
            model, tokenizer, features, question_text, benchmark, sample, device, conv_mode,
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

        if (sample_idx + 1) % 10 == 0:
            acc = correct_count / (sample_idx + 1) * 100
            logger.info(f"  Eval [{sample_idx + 1}/{total}]: acc={acc:.1f}%")

    accuracy = correct_count / total * 100 if total > 0 else 0

    # Save final universal embedding tensors
    torch.save(
        [f.detach().cpu() for f in features],
        os.path.join(tensors_dir, f"{benchmark.name}_universal.pt"),
    )

    # Write per-question eval results
    with open(results_file, "w") as f:
        for r in per_question_results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "mode": "universal",
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

    summary_file = os.path.join(results_dir, f"{benchmark.name}_universal_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Universal optimization complete: {summary}")
    return summary


def optimize_per_question(
    model,
    tokenizer,
    image_processor,
    benchmark: Benchmark,
    max_samples: int = 50,
    num_steps: int = 50,
    lr: float = 0.001,
    conv_mode: str = "llama_3",
    results_dir: str = "results/optimization",
    _skip_setup: bool = False,
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
        _skip_setup: If True, skip model setup (already done by universal).

    Returns:
        Summary dict with accuracy metrics.
    """
    os.makedirs(results_dir, exist_ok=True)

    if _skip_setup:
        device = next(model.parameters()).device
        dtype = torch.float32
        shapes = get_encoder_output_shapes(model, image_processor)
    else:
        shapes, device, dtype = _setup_model_for_optimization(model, image_processor)

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

    logger.info(f"Starting PER-QUESTION optimization: {total} questions, {num_steps} steps each, lr={lr}")
    logger.info(f"Results file: {results_file}")

    for sample_idx, sample in enumerate(samples):
        if sample.question_id in completed_ids:
            continue

        question_text = benchmark.format_question(sample)
        answer_text = sample.ground_truth

        # Initialize fresh features for this question
        features = _init_features(shapes, device, dtype)
        optimizer = torch.optim.Adam(features, lr=lr)

        # Measure initial loss and answer BEFORE optimization (random embeddings)
        with torch.no_grad(), encode_images_hook(model, features):
            initial_loss = compute_teacher_forcing_loss(
                model, tokenizer, question_text, answer_text,
                conv_mode=conv_mode,
            ).item()
        initial_response, initial_prediction, initial_correct = _check_answer(
            model, tokenizer, features, question_text, benchmark, sample, device, conv_mode,
        )

        # Optimization loop with per-step logging
        losses = []
        start_time = time.time()
        nan_detected = False

        for step in range(num_steps):
            optimizer.zero_grad()

            with encode_images_hook(model, features):
                loss = compute_teacher_forcing_loss(
                    model, tokenizer, question_text, answer_text,
                    conv_mode=conv_mode,
                )

            loss.backward()

            # Gradient clipping to prevent FP16 overflow in mixed-precision chain
            torch.nn.utils.clip_grad_norm_(features, max_norm=1.0)

            optimizer.step()
            loss_val = loss.item()

            # NaN detection — stop early, don't corrupt results
            if not torch.isfinite(torch.tensor(loss_val)):
                logger.warning(
                    f"  Q{sample_idx + 1}/{total} step {step + 1}: NaN/Inf loss, stopping early"
                )
                nan_detected = True
                break

            losses.append(loss_val)

            # Log every 10 steps within a question
            if (step + 1) % 10 == 0:
                logger.info(
                    f"  Q{sample_idx + 1}/{total} step {step + 1}/{num_steps}: "
                    f"loss {losses[-1]:.4f} (started at {initial_loss:.4f})"
                )

        opt_time = time.time() - start_time
        final_loss = losses[-1] if losses else float("nan")

        # Check accuracy AFTER optimization: single forward pass + argmax.
        response, prediction, correct = _check_answer(
            model, tokenizer, features, question_text, benchmark, sample, device, conv_mode,
        )
        if correct:
            correct_after += 1

        result = {
            "question_id": sample.question_id,
            "question": sample.question[:200],
            "ground_truth": answer_text,
            # Before optimization (random embeddings)
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

        # Save optimized embedding tensors (for later inversion / analysis)
        tensors_dir = os.path.join(results_dir, "tensors")
        os.makedirs(tensors_dir, exist_ok=True)
        torch.save(
            [f.detach().cpu() for f in features],
            os.path.join(tensors_dir, f"{sample.question_id}.pt"),
        )

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
        "mode": "per_question",
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

    summary_file = os.path.join(results_dir, f"{benchmark.name}_optimized_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Per-question optimization complete: {summary}")
    return summary
