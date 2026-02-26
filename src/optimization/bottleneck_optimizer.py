"""Bottleneck optimization for Cambrian-8B.

Replaces the entire vision pipeline (4 encoders + SVA projector) with K
learnable token embeddings (default K=1) in the LLM's native embedding
space (dim=4096).  These are repeated to fill the expected image token
count so the LLM sees the correct sequence length and positional encoding.

For each question, optimizes these tokens via gradient descent to minimise
teacher-forcing loss on the correct answer.  After optimization, analyses
the learned tokens to discover whether a consistent codebook emerges.

This module is self-contained — it does NOT use encode_images hooks or the
multimodal projector.  Instead it builds inputs_embeds directly.

Rich per-step data collection supports downstream graphing:
  - Loss curves (per question and aggregated by answer class)
  - Token norm and gradient norm evolution
  - Inter-token cosine similarity over time
  - Snapshot accuracy checks at regular intervals
  - LM head decoded "words" at each snapshot
  - Confusion matrices, PCA/t-SNE, pairwise similarity heatmaps
"""

import json
import os
import time
import logging
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from cambrian.mm_utils import tokenizer_image_token
from cambrian.constants import IMAGE_TOKEN_INDEX

from src.benchmarks.base import Benchmark
from src.model.inference import build_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image token count discovery and expansion
# ---------------------------------------------------------------------------

def discover_image_token_count(model, tokenizer, image_processor, conv_mode="llama_3"):
    """Discover how many image tokens the model normally inserts.

    Runs a single dummy forward pass through the full vision pipeline and
    compares the output sequence length to the input to determine how many
    tokens IMAGE_TOKEN_INDEX expands into.

    Must be called BEFORE offloading vision encoders to CPU.

    Returns:
        int: number of image tokens (e.g. 576).
    """
    from PIL import Image
    from cambrian.mm_utils import process_images

    device = next(model.parameters()).device
    dummy_image = Image.new("RGB", (384, 384), color=(128, 128, 128))

    image_tensor = process_images([dummy_image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(dtype=model.dtype, device=device) for t in image_tensor]
    else:
        image_tensor = [image_tensor.to(dtype=model.dtype, device=device)]

    prompt = build_prompt("test", conv_mode=conv_mode, include_image=True)
    input_ids = tokenizer_image_token(
        prompt=prompt,
        tokenizer=tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    num_placeholders = (input_ids == IMAGE_TOKEN_INDEX).sum().item()

    with torch.no_grad():
        output = model.forward(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=[(384, 384)],
        )

    output_seq_len = output.logits.shape[1]
    input_seq_len = input_ids.shape[1]
    image_token_count = output_seq_len - input_seq_len + num_placeholders

    logger.info(f"Discovered image token count: {image_token_count}")
    return image_token_count


def _expand_bottleneck(bottleneck_tokens, num_image_tokens):
    """Expand K bottleneck tokens to fill num_image_tokens positions.

    For K=1: repeats the single token N times (all positions identical).
    For K=2: first half = token 0, second half = token 1 (block repeat).
    General: repeat_interleave each token to fill N/K positions.

    Args:
        bottleneck_tokens: Learnable tensor [K, hidden_dim].
        num_image_tokens: Target number of positions to fill.

    Returns:
        Expanded tensor [num_image_tokens, hidden_dim].
    """
    K = bottleneck_tokens.shape[0]
    if num_image_tokens is None or num_image_tokens <= K:
        return bottleneck_tokens

    tokens_per_slot = num_image_tokens // K
    expanded = bottleneck_tokens.repeat_interleave(tokens_per_slot, dim=0)

    # Handle remainder (pad with last token)
    remainder = num_image_tokens - expanded.shape[0]
    if remainder > 0:
        expanded = torch.cat([expanded, expanded[-1:].expand(remainder, -1)], dim=0)

    return expanded


# ---------------------------------------------------------------------------
# Core: build inputs_embeds with a bottleneck replacing image tokens
# ---------------------------------------------------------------------------

def _build_bottleneck_inputs(
    model,
    tokenizer,
    question: str,
    answer: str,
    bottleneck_tokens: torch.Tensor,
    conv_mode: str = "llama_3",
    num_image_tokens: int | None = None,
):
    """Build inputs_embeds and labels with bottleneck tokens replacing images.

    Instead of going through encode_images -> mm_projector -> splice, we:
      1. Tokenize the prompt (contains IMAGE_TOKEN_INDEX placeholder)
      2. Embed all real text tokens via the LLM's embedding table
      3. Replace the IMAGE_TOKEN_INDEX position(s) with our bottleneck tokens
         (expanded to num_image_tokens positions via repeat)
      4. Append embedded answer + EOS tokens
      5. Build labels (-100 for everything except answer tokens)

    Args:
        model: Cambrian model.
        tokenizer: Tokenizer.
        question: Formatted question string.
        answer: Ground truth answer text.
        bottleneck_tokens: Learnable tensor [K, hidden_dim], float32.
        conv_mode: Conversation template name.
        num_image_tokens: Expand bottleneck to this many positions (for correct
            sequence length). None = use bottleneck tokens as-is.

    Returns:
        (inputs_embeds, labels) both shaped [1, seq_len, ...].
    """
    device = next(model.parameters()).device
    dtype = model.dtype

    prompt = build_prompt(question, conv_mode=conv_mode, include_image=True)

    # Tokenize — IMAGE_TOKEN_INDEX (-200) appears where image tokens go
    prompt_ids = tokenizer_image_token(
        prompt=prompt,
        tokenizer=tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )  # [seq_len]

    # Tokenize answer + EOS
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_ids = torch.tensor(answer_ids, dtype=torch.long)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        answer_ids = torch.cat([answer_ids, torch.tensor([eos_id])])

    # Get the embedding layer
    embed_layer = model.get_input_embeddings()

    # Split prompt_ids into segments around IMAGE_TOKEN_INDEX
    prompt_ids_list = prompt_ids.tolist()
    image_positions = [i for i, tid in enumerate(prompt_ids_list) if tid == IMAGE_TOKEN_INDEX]

    if not image_positions:
        # No image token found — just embed everything normally
        all_ids = torch.cat([prompt_ids, answer_ids])
        inputs_embeds = embed_layer(all_ids.to(device)).to(dtype)
    else:
        # Build embeddings segment by segment
        segments = []

        # Text before first image token
        before_ids = prompt_ids[:image_positions[0]]
        if len(before_ids) > 0:
            segments.append(embed_layer(before_ids.to(device)).to(dtype))

        # Bottleneck tokens — expanded to fill image region, then cast to model dtype
        expanded = _expand_bottleneck(bottleneck_tokens, num_image_tokens)
        segments.append(expanded.to(dtype))

        # Text after last image token
        after_ids = prompt_ids[image_positions[-1] + 1:]
        if len(after_ids) > 0:
            segments.append(embed_layer(after_ids.to(device)).to(dtype))

        # Answer tokens
        segments.append(embed_layer(answer_ids.to(device)).to(dtype))

        inputs_embeds = torch.cat(segments, dim=0)  # [total_len, hidden_dim]

    inputs_embeds = inputs_embeds.unsqueeze(0)  # [1, seq_len, hidden_dim]

    # Labels: -100 for prompt, real IDs for answer+EOS
    total_len = inputs_embeds.shape[1]
    answer_len = len(answer_ids)
    labels = torch.full((1, total_len), -100, dtype=torch.long, device=device)
    labels[0, total_len - answer_len:] = answer_ids.to(device)

    return inputs_embeds, labels


def _bottleneck_forward_loss(
    model,
    tokenizer,
    question: str,
    answer: str,
    bottleneck_tokens: torch.Tensor,
    conv_mode: str = "llama_3",
    num_image_tokens: int | None = None,
    train_expand: int | None = None,
) -> torch.Tensor:
    """Compute teacher-forcing loss with bottleneck tokens replacing images.

    For training, uses a shorter expansion (train_expand) to fit in GPU
    memory.  The full expansion (num_image_tokens) is only used for
    inference under no_grad.

    Returns:
        Scalar loss tensor (differentiable w.r.t. bottleneck_tokens).
    """
    # Use shorter expansion for training to save memory
    expand_to = train_expand if train_expand is not None else num_image_tokens
    inputs_embeds, labels = _build_bottleneck_inputs(
        model, tokenizer, question, answer, bottleneck_tokens, conv_mode,
        num_image_tokens=expand_to,
    )
    output = model.forward(inputs_embeds=inputs_embeds, labels=labels)
    return output.loss


def _bottleneck_check_answer(
    model,
    tokenizer,
    question: str,
    bottleneck_tokens: torch.Tensor,
    benchmark,
    sample,
    conv_mode: str = "llama_3",
    num_image_tokens: int | None = None,
):
    """Forward pass with bottleneck tokens, argmax to check answer."""
    device = next(model.parameters()).device
    dtype = model.dtype

    prompt = build_prompt(question, conv_mode=conv_mode, include_image=True)
    prompt_ids = tokenizer_image_token(
        prompt=prompt,
        tokenizer=tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )

    embed_layer = model.get_input_embeddings()
    prompt_ids_list = prompt_ids.tolist()
    image_positions = [i for i, tid in enumerate(prompt_ids_list) if tid == IMAGE_TOKEN_INDEX]

    if not image_positions:
        inputs_embeds = embed_layer(prompt_ids.to(device)).to(dtype).unsqueeze(0)
    else:
        segments = []
        before_ids = prompt_ids[:image_positions[0]]
        if len(before_ids) > 0:
            segments.append(embed_layer(before_ids.to(device)).to(dtype))
        expanded = _expand_bottleneck(bottleneck_tokens, num_image_tokens)
        segments.append(expanded.to(dtype))
        after_ids = prompt_ids[image_positions[-1] + 1:]
        if len(after_ids) > 0:
            segments.append(embed_layer(after_ids.to(device)).to(dtype))
        inputs_embeds = torch.cat(segments, dim=0).unsqueeze(0)

    with torch.no_grad():
        output = model.forward(inputs_embeds=inputs_embeds)
        next_token_id = output.logits[0, -1].argmax().item()
        response = tokenizer.decode([next_token_id]).strip()

    prediction = benchmark.extract_answer(response, sample)
    correct = benchmark.score(prediction, sample)
    return response, prediction, correct


def _decode_tokens_lm_head(tokens, model, tokenizer, top_k=1):
    """Project bottleneck tokens through LayerNorm + lm_head to decode words.

    Args:
        tokens: Tensor [num_tokens, hidden_dim].
        model: Cambrian model (for lm_head, final norm).
        tokenizer: For decoding token IDs to strings.
        top_k: Number of top tokens to return per position.

    Returns:
        List (one per token position) of lists of (word, prob) tuples.
    """
    device = next(model.parameters()).device
    inner = getattr(model, "model", model)
    final_norm = getattr(inner, "norm", None)
    lm_head = model.lm_head

    results = []
    with torch.no_grad():
        for tok_idx in range(tokens.shape[0]):
            h = tokens[tok_idx].to(device=device, dtype=model.dtype).unsqueeze(0)
            if final_norm is not None:
                h = final_norm(h)
            logits = lm_head(h)  # [1, vocab_size]
            probs = torch.softmax(logits.float(), dim=-1).squeeze(0)
            top_probs, top_ids = probs.topk(top_k)
            decoded = [
                (tokenizer.decode([top_ids[k].item()]), round(top_probs[k].item(), 6))
                for k in range(top_k)
            ]
            results.append(decoded)
    return results


# ---------------------------------------------------------------------------
# Per-question optimiser with rich data collection
# ---------------------------------------------------------------------------

def optimize_bottleneck_per_question(
    model,
    tokenizer,
    benchmark: Benchmark,
    image_processor=None,
    num_tokens: int = 1,
    max_samples: int = 50,
    num_steps: int = 50,
    lr: float = 0.01,
    train_expand: int = 1,
    conv_mode: str = "llama_3",
    results_dir: str = "results/bottleneck",
    snapshot_every: int = 5,
) -> dict:
    """Optimize a narrow bottleneck independently for each question.

    Collects rich per-step metrics for downstream graphing:
      - Loss at every step
      - Token L2 norms at every step (per token)
      - Gradient L2 norms at every step (per token)
      - Cosine similarity between the 2 tokens at every step
      - At snapshot intervals: accuracy check + LM head decoding

    Args:
        model: Loaded Cambrian model.
        tokenizer: Tokenizer.
        benchmark: Loaded benchmark instance.
        image_processor: Image processor (needed to discover image token count).
        num_tokens: Number of bottleneck tokens (default 1).
        max_samples: Number of questions to optimise.
        num_steps: Gradient descent steps per question.
        lr: Adam learning rate.
        train_expand: Number of token copies for training forward/backward pass.
            Kept small (default 16) to fit in GPU memory.  The full expansion
            (num_image_tokens) is only used for inference (answer checking)
            which runs under no_grad.
        conv_mode: Conversation template.
        results_dir: Where to save results.
        snapshot_every: Check accuracy + decode tokens every N steps.

    Returns:
        Summary dict with accuracy metrics.
    """
    os.makedirs(results_dir, exist_ok=True)
    tensors_dir = os.path.join(results_dir, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)

    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size

    # Discover image token count BEFORE offloading vision encoders
    num_image_tokens = None
    if image_processor is not None:
        num_image_tokens = discover_image_token_count(
            model, tokenizer, image_processor, conv_mode=conv_mode,
        )
        logger.info(
            f"Discovered normal image region = {num_image_tokens} tokens. "
            f"Using train_expand={train_expand} for all forward passes "
            f"(GPU memory constraint)."
        )
    else:
        logger.warning(
            "No image_processor provided — bottleneck tokens will NOT be "
            "expanded to match image region length. Positional encoding confound!"
        )

    # Offload vision encoders to save memory — we don't use them at all
    inner = getattr(model, "model", model)
    towers = getattr(inner, "vision_tower_aux_list", None)
    if towers:
        for tower in towers:
            tower.cpu()
        torch.cuda.empty_cache()
        logger.info(f"Offloaded {len(towers)} vision encoders to CPU")

    # Enable gradient checkpointing
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled")

    samples = list(benchmark)
    if max_samples:
        samples = samples[:max_samples]
    total = len(samples)

    results_file = os.path.join(results_dir, f"{benchmark.name}_bottleneck.jsonl")

    # Resume support
    completed_ids = set()
    results = []
    correct_after = 0
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

    run_start = time.time()
    logger.info(
        f"Starting BOTTLENECK per-question optimization: {total} questions, "
        f"{num_tokens} tokens x {hidden_dim} dim, {num_steps} steps, lr={lr}, "
        f"snapshots every {snapshot_every} steps"
    )

    for sample_idx, sample in enumerate(samples):
        if sample.question_id in completed_ids:
            continue

        question_text = benchmark.format_question(sample)
        answer_text = sample.ground_truth

        # Initialise random bottleneck tokens
        tokens = torch.randn(
            num_tokens, hidden_dim,
            device=device, dtype=torch.float32,
        ) * 0.02
        tokens.requires_grad_(True)
        optimizer = torch.optim.Adam([tokens], lr=lr)

        # Save initial tokens
        initial_tokens_cpu = tokens.detach().cpu().clone()
        torch.save(
            initial_tokens_cpu,
            os.path.join(tensors_dir, f"{sample.question_id}_initial.pt"),
        )

        # Initial measurements (before any optimisation)
        # Use train_expand for all forward passes to stay within GPU memory
        with torch.no_grad():
            initial_loss = _bottleneck_forward_loss(
                model, tokenizer, question_text, answer_text, tokens, conv_mode,
                num_image_tokens=num_image_tokens,
                train_expand=train_expand,
            ).item()
        initial_response, initial_pred, initial_correct = _bottleneck_check_answer(
            model, tokenizer, question_text, tokens, benchmark, sample, conv_mode,
            num_image_tokens=train_expand,
        )
        initial_lm_decode = _decode_tokens_lm_head(tokens.detach(), model, tokenizer, top_k=5)
        initial_token_norms = [round(tokens[i].detach().norm().item(), 6) for i in range(num_tokens)]

        # Per-step tracking arrays
        losses = []
        token_norms = [[] for _ in range(num_tokens)]  # token_norms[tok_idx][step]
        grad_norms = [[] for _ in range(num_tokens)]
        inter_token_cosine = []
        snapshots = []

        # Record step-0 snapshot (initial state)
        snapshots.append({
            "step": 0,
            "loss": round(initial_loss, 4),
            "prediction": initial_pred,
            "correct": initial_correct,
            "lm_head_top1": [
                [d[0][0], d[0][1]] for d in initial_lm_decode  # (word, prob) per token
            ],
            "token_norms": initial_token_norms,
        })

        # Optimisation loop
        start_time = time.time()
        nan_detected = False

        for step in range(num_steps):
            optimizer.zero_grad()
            loss = _bottleneck_forward_loss(
                model, tokenizer, question_text, answer_text, tokens, conv_mode,
                num_image_tokens=num_image_tokens,
                train_expand=train_expand,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([tokens], max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            if not torch.isfinite(torch.tensor(loss_val)):
                logger.warning(f"  Q{sample_idx+1}/{total} step {step+1}: NaN, stopping")
                nan_detected = True
                break
            losses.append(loss_val)

            # --- Per-step cheap metrics (negligible cost) ---
            with torch.no_grad():
                for ti in range(num_tokens):
                    token_norms[ti].append(round(tokens[ti].norm().item(), 6))
                    if tokens.grad is not None:
                        grad_norms[ti].append(round(tokens.grad[ti].norm().item(), 6))
                    else:
                        grad_norms[ti].append(0.0)

                if num_tokens >= 2:
                    cos = F.cosine_similarity(
                        tokens[0].unsqueeze(0), tokens[1].unsqueeze(0)
                    ).item()
                    inter_token_cosine.append(round(cos, 6))

            # --- Snapshot (accuracy + LM head decode, every N steps) ---
            is_snapshot = ((step + 1) % snapshot_every == 0) or (step == num_steps - 1)
            if is_snapshot:
                snap_response, snap_pred, snap_correct = _bottleneck_check_answer(
                    model, tokenizer, question_text, tokens, benchmark, sample, conv_mode,
                    num_image_tokens=train_expand,
                )
                snap_decode = _decode_tokens_lm_head(tokens.detach(), model, tokenizer, top_k=5)
                snap_norms = [round(tokens[i].detach().norm().item(), 6) for i in range(num_tokens)]

                snapshots.append({
                    "step": step + 1,
                    "loss": round(loss_val, 4),
                    "prediction": snap_pred,
                    "correct": snap_correct,
                    "lm_head_top1": [
                        [d[0][0], d[0][1]] for d in snap_decode
                    ],
                    "lm_head_top5": [
                        [[w, p] for w, p in d] for d in snap_decode
                    ],
                    "token_norms": snap_norms,
                })

            if (step + 1) % 10 == 0:
                logger.info(
                    f"  Q{sample_idx+1}/{total} step {step+1}/{num_steps}: "
                    f"loss {loss_val:.4f} (was {initial_loss:.4f})"
                )

        opt_time = time.time() - start_time
        final_loss = losses[-1] if losses else float("nan")

        # Final answer check
        response, prediction, correct = _bottleneck_check_answer(
            model, tokenizer, question_text, tokens, benchmark, sample, conv_mode,
            num_image_tokens=train_expand,
        )
        if correct:
            correct_after += 1

        # Final token stats
        final_token_norms = [round(tokens[i].detach().norm().item(), 6) for i in range(num_tokens)]
        final_lm_decode = _decode_tokens_lm_head(tokens.detach(), model, tokenizer, top_k=5)

        # Save final tokens
        torch.save(
            tokens.detach().cpu(),
            os.path.join(tensors_dir, f"{sample.question_id}.pt"),
        )

        result = {
            "question_id": sample.question_id,
            "question": sample.question[:200],
            "ground_truth": answer_text,
            "choices": sample.choices,
            "num_answer_choices": len(sample.choices) if sample.choices else None,
            "num_tokens": num_tokens,
            "num_image_tokens": num_image_tokens,

            # Before optimisation
            "initial_prediction": initial_pred,
            "initial_correct": initial_correct,
            "initial_loss": round(initial_loss, 4),
            "initial_token_norms": initial_token_norms,
            "initial_lm_decode": [
                [[w, p] for w, p in d] for d in initial_lm_decode
            ],

            # After optimisation
            "prediction": prediction,
            "raw_response": response[:200],
            "correct": correct,
            "final_loss": round(final_loss, 4) if not nan_detected else None,
            "loss_reduction": round(initial_loss - final_loss, 4) if not nan_detected else None,
            "final_token_norms": final_token_norms,
            "final_lm_decode": [
                [[w, p] for w, p in d] for d in final_lm_decode
            ],

            # Per-step curves (every step)
            "loss_curve": [round(l, 4) for l in losses],
            "token_norms": [[round(v, 4) for v in tn] for tn in token_norms],
            "grad_norms": [[round(v, 4) for v in gn] for gn in grad_norms],
            "inter_token_cosine": [round(v, 4) for v in inter_token_cosine],

            # Snapshots (every N steps: accuracy + LM decode)
            "snapshots": snapshots,

            # Metadata
            "optimization_time_s": round(opt_time, 1),
            "num_steps": len(losses),
            "nan_detected": nan_detected,
        }
        results.append(result)

        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        done = len(results)
        acc = correct_after / done * 100
        elapsed = time.time() - run_start
        remaining = total - done
        active = done - len(completed_ids)
        avg_time = elapsed / active if active > 0 else 0
        eta = remaining * avg_time

        status = "CORRECT" if correct else "WRONG"
        loss_str = (
            f"{initial_loss:.3f}->{final_loss:.3f}"
            if not nan_detected else f"{initial_loss:.3f}->NaN"
        )
        logger.info(
            f"[{done}/{total}] {sample.question_id}: {status} "
            f"(pred={prediction}, gt={answer_text}) "
            f"loss {loss_str} | acc={acc:.1f}% | {opt_time:.1f}s | ETA {eta/60:.0f}min"
        )

    # Summary
    accuracy = correct_after / len(results) * 100 if results else 0
    valid = [r for r in results if r.get("loss_reduction") is not None]
    nan_count = sum(1 for r in results if r.get("nan_detected"))

    summary = {
        "mode": "bottleneck_per_question",
        "benchmark": benchmark.name,
        "num_tokens": num_tokens,
        "num_image_tokens": num_image_tokens,
        "num_samples": len(results),
        "num_nan": nan_count,
        "num_steps": num_steps,
        "learning_rate": lr,
        "snapshot_every": snapshot_every,
        "accuracy_after_optimization": round(accuracy, 2),
        "accuracy_before_optimization": round(
            sum(r["initial_correct"] for r in results) / len(results) * 100, 2
        ) if results else 0,
        "avg_initial_loss": round(
            sum(r["initial_loss"] for r in results) / len(results), 4
        ) if results else 0,
        "avg_final_loss": round(
            sum(r["final_loss"] for r in valid) / len(valid), 4
        ) if valid else 0,
    }

    summary_file = os.path.join(results_dir, f"{benchmark.name}_bottleneck_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Bottleneck optimization complete: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Codebook analysis — produces graph-ready data
# ---------------------------------------------------------------------------

def analyse_codebook(
    results_dir: str,
    benchmark_name: str,
    model=None,
    tokenizer=None,
    top_k: int = 10,
) -> dict:
    """Analyse the learned bottleneck tokens to discover codebook structure.

    Produces graph-ready data including:
      - Per-answer-class statistics and accuracy
      - Between-class centroid distances (L2, cosine)
      - Confusion matrix
      - PCA and t-SNE projections (with initial token trajectories)
      - Aggregated loss/norm/cosine curves by answer class (mean + std)
      - Step-accuracy evolution (what fraction correct at each snapshot step)
      - Pairwise cosine similarity between all samples
      - LM head decoded words per answer class centroid
      - Per-answer final loss distributions (for box/violin plots)

    Args:
        results_dir: Directory containing bottleneck results and tensors/.
        benchmark_name: Benchmark name (for filenames).
        model: Optional loaded model (needed for LM head decoding).
        tokenizer: Optional tokenizer (needed for LM head decoding).
        top_k: Number of top decoded tokens to report per position.

    Returns:
        Analysis dict (also saved to JSON).
    """
    tensors_dir = os.path.join(results_dir, "tensors")
    results_file = os.path.join(results_dir, f"{benchmark_name}_bottleneck.jsonl")

    # --- Load all results and tensors ---
    all_results = []
    results_by_answer = defaultdict(list)
    all_vectors = []         # final token vectors (flattened)
    all_initial_vectors = [] # initial token vectors (flattened)
    all_answers = []
    all_qids = []
    all_correct = []

    with open(results_file) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r["question_id"]
            gt = r["ground_truth"]
            tensor_path = os.path.join(tensors_dir, f"{qid}.pt")
            initial_path = os.path.join(tensors_dir, f"{qid}_initial.pt")
            if not os.path.exists(tensor_path):
                continue

            tokens = torch.load(tensor_path, map_location="cpu", weights_only=True)
            flat = tokens.flatten().numpy()

            initial_flat = None
            if os.path.exists(initial_path):
                initial_tokens = torch.load(initial_path, map_location="cpu", weights_only=True)
                initial_flat = initial_tokens.flatten().numpy()

            entry = {
                "question_id": qid,
                "tokens": tokens,
                "flat": flat,
                "initial_flat": initial_flat,
                "correct": r.get("correct", False),
                "prediction": r.get("prediction", ""),
                "result": r,
            }
            results_by_answer[gt].append(entry)
            all_results.append(r)
            all_vectors.append(flat)
            if initial_flat is not None:
                all_initial_vectors.append(initial_flat)
            all_answers.append(gt)
            all_qids.append(qid)
            all_correct.append(r.get("correct", False))

    if not all_vectors:
        logger.warning("No bottleneck tensors found")
        return {}

    all_vectors = np.array(all_vectors)
    has_initials = len(all_initial_vectors) == len(all_vectors)
    if has_initials:
        all_initial_vectors = np.array(all_initial_vectors)

    logger.info(
        f"Loaded {len(all_vectors)} token pairs across "
        f"{len(results_by_answer)} answer classes"
    )

    analysis = {
        "num_samples": len(all_vectors),
        "num_correct": sum(all_correct),
        "accuracy": round(sum(all_correct) / len(all_correct) * 100, 2),
    }

    # ------------------------------------------------------------------
    # 1. Per-answer-class statistics
    # ------------------------------------------------------------------
    answer_classes = {}
    class_centroids = {}

    for answer, items in sorted(results_by_answer.items()):
        vecs = np.array([it["flat"] for it in items])
        acc = sum(it["correct"] for it in items) / len(items) * 100
        centroid = vecs.mean(axis=0)
        dists = np.linalg.norm(vecs - centroid, axis=1)

        class_centroids[answer] = centroid
        answer_classes[answer] = {
            "count": len(items),
            "accuracy": round(acc, 1),
            "within_class_spread_mean": round(float(dists.mean()), 4),
            "within_class_spread_std": round(float(dists.std()), 4),
        }
        logger.info(
            f"  Answer '{answer}': n={len(items)}, acc={acc:.1f}%, "
            f"spread={dists.mean():.2f}+-{dists.std():.2f}"
        )
    analysis["answer_classes"] = answer_classes

    # ------------------------------------------------------------------
    # 2. Between-class centroid distances
    # ------------------------------------------------------------------
    answers_sorted = sorted(class_centroids.keys())
    between_distances = {}
    for i, a1 in enumerate(answers_sorted):
        for a2 in answers_sorted[i + 1:]:
            l2 = float(np.linalg.norm(class_centroids[a1] - class_centroids[a2]))
            cos = float(
                np.dot(class_centroids[a1], class_centroids[a2])
                / (np.linalg.norm(class_centroids[a1]) * np.linalg.norm(class_centroids[a2]) + 1e-8)
            )
            between_distances[f"{a1}_vs_{a2}"] = {
                "l2_distance": round(l2, 4),
                "cosine_similarity": round(cos, 6),
            }
            logger.info(f"  {a1} vs {a2}: L2={l2:.2f}, cos={cos:.4f}")
    analysis["between_class_distances"] = between_distances

    # ------------------------------------------------------------------
    # 3. Confusion matrix
    # ------------------------------------------------------------------
    confusion = defaultdict(lambda: defaultdict(int))
    for r in all_results:
        gt = r["ground_truth"]
        pred = r.get("prediction", "")
        confusion[gt][pred] += 1
    # Convert to regular dict for JSON
    analysis["confusion_matrix"] = {
        gt: dict(preds) for gt, preds in sorted(confusion.items())
    }
    logger.info(f"Confusion matrix: {dict(analysis['confusion_matrix'])}")

    # ------------------------------------------------------------------
    # 4. PCA projection (final + initial for trajectories)
    # ------------------------------------------------------------------
    if len(all_vectors) >= 3:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(3, len(all_vectors)))
        coords = pca.fit_transform(all_vectors)

        pca_data = {
            "explained_variance_ratio": [
                round(v, 4) for v in pca.explained_variance_ratio_.tolist()
            ],
            "points": [
                {
                    "question_id": all_qids[i],
                    "answer": all_answers[i],
                    "correct": all_correct[i],
                    "x": round(float(coords[i, 0]), 4),
                    "y": round(float(coords[i, 1]), 4),
                    "z": round(float(coords[i, 2]), 4) if coords.shape[1] > 2 else 0.0,
                }
                for i in range(len(coords))
            ],
        }

        # Project initial tokens into same PCA space for trajectory arrows
        if has_initials:
            initial_coords = pca.transform(all_initial_vectors)
            pca_data["initial_points"] = [
                {
                    "question_id": all_qids[i],
                    "answer": all_answers[i],
                    "x": round(float(initial_coords[i, 0]), 4),
                    "y": round(float(initial_coords[i, 1]), 4),
                    "z": round(float(initial_coords[i, 2]), 4) if initial_coords.shape[1] > 2 else 0.0,
                }
                for i in range(len(initial_coords))
            ]

        analysis["pca"] = pca_data
        logger.info(f"PCA variance: {pca_data['explained_variance_ratio']}")

    # ------------------------------------------------------------------
    # 5. t-SNE projection
    # ------------------------------------------------------------------
    if len(all_vectors) >= 5:
        try:
            from sklearn.manifold import TSNE

            perplexity = min(30, len(all_vectors) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            tsne_coords = tsne.fit_transform(all_vectors)

            analysis["tsne"] = {
                "perplexity": perplexity,
                "points": [
                    {
                        "question_id": all_qids[i],
                        "answer": all_answers[i],
                        "correct": all_correct[i],
                        "x": round(float(tsne_coords[i, 0]), 4),
                        "y": round(float(tsne_coords[i, 1]), 4),
                    }
                    for i in range(len(tsne_coords))
                ],
            }
            logger.info("t-SNE projection computed")
        except Exception as e:
            logger.warning(f"t-SNE failed: {e}")

    # ------------------------------------------------------------------
    # 6. Pairwise cosine similarity matrix (for heatmap)
    # ------------------------------------------------------------------
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-8
    normed = all_vectors / norms
    cos_matrix = normed @ normed.T  # [N, N]

    # Sort by answer class for block-diagonal structure
    sort_idx = sorted(range(len(all_answers)), key=lambda i: all_answers[i])
    cos_sorted = cos_matrix[np.ix_(sort_idx, sort_idx)]

    analysis["pairwise_cosine"] = {
        "order": [all_qids[i] for i in sort_idx],
        "answer_order": [all_answers[i] for i in sort_idx],
        "matrix": [[round(float(v), 4) for v in row] for row in cos_sorted.tolist()],
    }
    logger.info(f"Pairwise cosine similarity matrix: {cos_sorted.shape}")

    # ------------------------------------------------------------------
    # 7. Aggregated loss curves by answer class (mean + std at each step)
    # ------------------------------------------------------------------
    loss_curves_by_answer = {}
    for answer, items in sorted(results_by_answer.items()):
        curves = [it["result"].get("loss_curve", []) for it in items]
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        # Pad shorter curves with their last value
        padded = []
        for c in curves:
            if len(c) < max_len and c:
                c = c + [c[-1]] * (max_len - len(c))
            padded.append(c)
        arr = np.array(padded)
        loss_curves_by_answer[answer] = {
            "mean": [round(float(v), 4) for v in arr.mean(axis=0)],
            "std": [round(float(v), 4) for v in arr.std(axis=0)],
            "count": len(padded),
        }
    analysis["loss_curves_by_answer"] = loss_curves_by_answer

    # ------------------------------------------------------------------
    # 8. Per-answer final loss distributions (for box/violin plots)
    # ------------------------------------------------------------------
    final_losses_by_answer = {}
    for answer, items in sorted(results_by_answer.items()):
        losses = [
            it["result"]["final_loss"]
            for it in items
            if it["result"].get("final_loss") is not None
        ]
        final_losses_by_answer[answer] = losses
    analysis["final_loss_distributions"] = final_losses_by_answer

    # ------------------------------------------------------------------
    # 9. Step-accuracy curve (aggregated across all questions)
    # ------------------------------------------------------------------
    # Collect all snapshot steps across all questions
    step_correct = defaultdict(list)  # step -> list of booleans
    for r in all_results:
        for snap in r.get("snapshots", []):
            step_correct[snap["step"]].append(snap["correct"])

    step_accuracy_curve = []
    for step in sorted(step_correct.keys()):
        bools = step_correct[step]
        step_accuracy_curve.append({
            "step": step,
            "num_samples": len(bools),
            "num_correct": sum(bools),
            "accuracy": round(sum(bools) / len(bools) * 100, 2),
        })
    analysis["step_accuracy_curve"] = step_accuracy_curve
    if step_accuracy_curve:
        logger.info(
            f"Step-accuracy: step 0 = {step_accuracy_curve[0]['accuracy']:.1f}% → "
            f"step {step_accuracy_curve[-1]['step']} = {step_accuracy_curve[-1]['accuracy']:.1f}%"
        )

    # ------------------------------------------------------------------
    # 10. Token norm evolution by answer class (mean + std)
    # ------------------------------------------------------------------
    norm_curves_by_answer = {}
    for answer, items in sorted(results_by_answer.items()):
        all_tn = [it["result"].get("token_norms", []) for it in items]
        if not all_tn or not all_tn[0]:
            continue
        num_toks = len(all_tn[0])
        per_tok = {}
        for ti in range(num_toks):
            curves = [tn[ti] for tn in all_tn if ti < len(tn)]
            if not curves:
                continue
            max_len = max(len(c) for c in curves)
            padded = []
            for c in curves:
                if len(c) < max_len and c:
                    c = c + [c[-1]] * (max_len - len(c))
                padded.append(c)
            arr = np.array(padded)
            per_tok[f"token_{ti}_mean"] = [round(float(v), 4) for v in arr.mean(axis=0)]
            per_tok[f"token_{ti}_std"] = [round(float(v), 4) for v in arr.std(axis=0)]
        norm_curves_by_answer[answer] = per_tok
    analysis["token_norm_evolution_by_answer"] = norm_curves_by_answer

    # ------------------------------------------------------------------
    # 11. Inter-token cosine evolution by answer class
    # ------------------------------------------------------------------
    cosine_curves_by_answer = {}
    for answer, items in sorted(results_by_answer.items()):
        curves = [it["result"].get("inter_token_cosine", []) for it in items]
        curves = [c for c in curves if c]
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = []
        for c in curves:
            if len(c) < max_len:
                c = c + [c[-1]] * (max_len - len(c))
            padded.append(c)
        arr = np.array(padded)
        cosine_curves_by_answer[answer] = {
            "mean": [round(float(v), 4) for v in arr.mean(axis=0)],
            "std": [round(float(v), 4) for v in arr.std(axis=0)],
        }
    analysis["inter_token_cosine_by_answer"] = cosine_curves_by_answer

    # ------------------------------------------------------------------
    # 12. LM head decoding of class centroids
    # ------------------------------------------------------------------
    if model is not None and tokenizer is not None:
        logger.info("Decoding class centroid tokens through LM head...")
        decoding_by_answer = {}
        for answer, items in sorted(results_by_answer.items()):
            all_tokens = torch.stack([it["tokens"] for it in items])
            centroid_tokens = all_tokens.float().mean(dim=0)
            decoded = _decode_tokens_lm_head(centroid_tokens, model, tokenizer, top_k=top_k)
            decoding_by_answer[answer] = [
                [{"token": w, "prob": p} for w, p in d] for d in decoded
            ]
            top_str = ", ".join(f"'{d[0][0]}'({d[0][1]:.3f})" for d in decoded)
            logger.info(f"  Answer '{answer}' centroid -> [{top_str}]")

        # Also decode EVERY individual question's tokens (for scatter plot colouring)
        per_sample_decoding = []
        for r, vec, ans, qid in zip(all_results, all_vectors, all_answers, all_qids):
            tensor_path = os.path.join(tensors_dir, f"{qid}.pt")
            tokens = torch.load(tensor_path, map_location="cpu", weights_only=True)
            decoded = _decode_tokens_lm_head(tokens, model, tokenizer, top_k=3)
            per_sample_decoding.append({
                "question_id": qid,
                "answer": ans,
                "correct": r.get("correct", False),
                "decoded_tokens": [
                    [{"token": w, "prob": p} for w, p in d] for d in decoded
                ],
            })

        analysis["lm_head_decoding"] = decoding_by_answer
        analysis["per_sample_lm_decoding"] = per_sample_decoding

    # ------------------------------------------------------------------
    # 13. Snapshot LM head evolution (how decoded words change over training)
    # ------------------------------------------------------------------
    # Aggregate snapshot lm_head_top1 by step and answer
    lm_evolution_by_answer = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        gt = r["ground_truth"]
        for snap in r.get("snapshots", []):
            step = snap["step"]
            top1 = snap.get("lm_head_top1", [])
            if top1:
                lm_evolution_by_answer[gt][step].append(top1)

    # For each answer class and step, find the most common decoded word per token position
    lm_evolution_summary = {}
    for answer in sorted(lm_evolution_by_answer.keys()):
        steps_data = {}
        for step in sorted(lm_evolution_by_answer[answer].keys()):
            entries = lm_evolution_by_answer[answer][step]
            if not entries:
                continue
            num_toks = len(entries[0])
            per_tok = []
            for ti in range(num_toks):
                words = [e[ti][0] for e in entries if ti < len(e)]
                from collections import Counter
                counts = Counter(words)
                most_common = counts.most_common(3)
                per_tok.append([
                    {"token": w, "count": c, "fraction": round(c / len(words), 3)}
                    for w, c in most_common
                ])
            steps_data[str(step)] = per_tok
        lm_evolution_summary[answer] = steps_data
    analysis["lm_head_evolution_by_answer"] = lm_evolution_summary

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    analysis_file = os.path.join(results_dir, f"{benchmark_name}_codebook_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Codebook analysis saved to {analysis_file}")

    return analysis
