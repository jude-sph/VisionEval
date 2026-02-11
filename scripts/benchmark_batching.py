"""Benchmark batched forward passes through Cambrian-8B.

Tests whether Cambrian supports batch_size > 1, finds the max batch size
before OOM, and measures throughput at each batch size.

Run this BEFORE implementing batched optimization to validate assumptions.

Usage:
    python scripts/benchmark_batching.py
    python scripts/benchmark_batching.py --gpu_ids 0 --batch_sizes 1,2,4
"""

import os
import sys
import time
import logging
import torch
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_batching")


def _left_pad_sequences(sequences: list[list[int]], pad_token_id: int):
    """Left-pad a list of token ID lists to the same length.

    Returns:
        input_ids: [batch, max_len] LongTensor
        attention_mask: [batch, max_len] LongTensor (1=real, 0=pad)
    """
    max_len = max(len(s) for s in sequences)
    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        pad_len = max_len - len(seq)
        input_ids[i, pad_len:] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, pad_len:] = 1
    return input_ids, attention_mask


def main(
    gpu_ids: str = "0,1,2,3",
    model_path: str = "nyu-visionx/cambrian-8b",
    benchmark: str = "mmmu",
    max_samples: int = 16,
    batch_sizes: str = "1,2,4,8,16",
    conv_mode: str = "llama_3",
    num_warmup: int = 1,
    num_trials: int = 3,
):
    """Benchmark batched forward passes.

    Args:
        gpu_ids: Comma-separated GPU indices.
        model_path: HuggingFace model path.
        benchmark: Benchmark to load questions from.
        max_samples: Number of questions to load (must be >= max batch_size).
        batch_sizes: Comma-separated batch sizes to test.
        conv_mode: Conversation template.
        num_warmup: Warmup iterations before timing.
        num_trials: Timed iterations to average.
    """
    # Parse inputs
    if isinstance(gpu_ids, (list, tuple)):
        gpu_list = [int(x) for x in gpu_ids]
    else:
        gpu_list = [int(x) for x in str(gpu_ids).split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    remapped_gpus = list(range(len(gpu_list)))

    if isinstance(batch_sizes, (list, tuple)):
        bs_list = [int(x) for x in batch_sizes]
    else:
        bs_list = [int(x) for x in str(batch_sizes).split(",")]

    # Load model
    from src.model.loader import load_cambrian
    logger.info(f"Loading model on GPUs {gpu_list}...")
    tokenizer, model, image_processor, context_len = load_cambrian(
        model_path=model_path,
        gpu_ids=remapped_gpus,
    )
    device = next(model.parameters()).device
    logger.info(f"Model loaded on {device}")

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token_id = eos_token_id = {tokenizer.pad_token_id}")

    # Load benchmark questions
    from src.benchmarks import get_benchmark
    bench = get_benchmark(benchmark)
    bench.load(max_samples=max_samples)
    samples = list(bench)
    logger.info(f"Loaded {len(samples)} samples from {benchmark}")

    # Setup model for embedding optimization (offload encoders, enable checkpointing)
    from src.optimization.utils import encode_images_hook, get_encoder_output_shapes

    shapes = get_encoder_output_shapes(model, image_processor)
    inner = getattr(model, "model", model)
    towers = getattr(inner, "vision_tower_aux_list", None)
    if towers:
        for tower in towers:
            tower.cpu()
        torch.cuda.empty_cache()
        logger.info(f"Moved {len(towers)} vision encoders to CPU")

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Create learnable features (universal noise)
    features = []
    for shape in shapes:
        t = torch.randn(shape, device=device, dtype=torch.float32) * 0.02
        t.requires_grad_(True)
        features.append(t)

    # Pre-tokenize all questions
    from cambrian.mm_utils import tokenizer_image_token
    from cambrian.constants import IMAGE_TOKEN_INDEX
    from src.model.inference import build_prompt

    all_prompt_seqs = []  # prompt only (for inference)
    all_full_seqs = []    # prompt + answer + eos (for training)
    all_prompt_lens = []

    for sample in samples:
        question_text = bench.format_question(sample)
        answer_text = sample.ground_truth

        prompt = build_prompt(question_text, conv_mode=conv_mode, include_image=True)
        prompt_ids = tokenizer_image_token(
            prompt=prompt,
            tokenizer=tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).tolist()  # 1D list

        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        eos_id = tokenizer.eos_token_id
        full_ids = prompt_ids + answer_ids + ([eos_id] if eos_id else [])

        all_prompt_seqs.append(prompt_ids)
        all_full_seqs.append(full_ids)
        all_prompt_lens.append(len(prompt_ids))

    logger.info(f"Tokenized {len(all_prompt_seqs)} questions")
    logger.info(f"Prompt lengths: min={min(all_prompt_lens)}, max={max(all_prompt_lens)}, "
                f"mean={sum(all_prompt_lens)/len(all_prompt_lens):.0f}")

    # Baseline: single-sample forward (batch_size=1) for correctness reference
    logger.info("=" * 70)
    logger.info("Running single-sample baseline for correctness reference...")
    baseline_logits_last = []
    baseline_losses = []

    for i in range(min(4, len(samples))):
        ids = torch.tensor(all_full_seqs[i], dtype=torch.long).unsqueeze(0).to(device)
        labels = ids.clone()
        labels[0, :all_prompt_lens[i]] = -100

        with encode_images_hook(model, features):
            output = model.forward(
                input_ids=ids,
                labels=labels,
                images=[torch.zeros(1, 3, 384, 384, device=device, dtype=model.dtype) for _ in range(4)],
                image_sizes=[(384, 384)],
            )
        baseline_losses.append(output.loss.item())
        baseline_logits_last.append(output.logits[0, -1].detach().cpu())
        logger.info(f"  Sample {i}: loss={output.loss.item():.4f}, "
                    f"top token='{tokenizer.decode([output.logits[0, -1].argmax().item()])}'")

    logger.info(f"Baseline losses: {[round(l, 4) for l in baseline_losses]}")

    # Benchmark each batch size
    results = []
    max_inference_bs = 0
    max_training_bs = 0

    for bs in bs_list:
        if bs > len(samples):
            logger.warning(f"Batch size {bs} > {len(samples)} samples, skipping")
            continue

        logger.info("=" * 70)
        logger.info(f"TESTING BATCH SIZE = {bs}")
        logger.info("=" * 70)

        result = {"batch_size": bs}

        # --- Inference test ---
        logger.info(f"  [Inference] batch_size={bs}...")
        try:
            # Left-pad prompt-only sequences
            batch_seqs = all_prompt_seqs[:bs]
            input_ids, attention_mask = _left_pad_sequences(batch_seqs, tokenizer.pad_token_id)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            def patched_encode(image_aux_list, *args, **kwargs):
                return [f.to(model.dtype).expand(bs, -1, -1) for f in features]

            original_fn = model.encode_images
            model.encode_images = patched_encode

            # Warmup
            for _ in range(num_warmup):
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    output = model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=[torch.zeros(bs, 3, 384, 384, device=device, dtype=model.dtype) for _ in range(4)],
                        image_sizes=[(384, 384)] * bs,
                    )

            # Timed trials
            torch.cuda.synchronize()
            times = []
            for _ in range(num_trials):
                torch.cuda.reset_peak_memory_stats()
                start = time.time()
                with torch.no_grad():
                    output = model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=[torch.zeros(bs, 3, 384, 384, device=device, dtype=model.dtype) for _ in range(4)],
                        image_sizes=[(384, 384)] * bs,
                    )
                torch.cuda.synchronize()
                times.append(time.time() - start)

            model.encode_images = original_fn

            avg_time = sum(times) / len(times)
            peak_mem = max(torch.cuda.max_memory_allocated(i) for i in range(len(remapped_gpus))) / 1e9
            throughput = bs / avg_time

            # Verify: logits[:, -1] should give valid predictions
            last_logits = output.logits[:, -1]
            top_tokens = [tokenizer.decode([last_logits[i].argmax().item()]) for i in range(bs)]

            result["inference_time_s"] = round(avg_time, 3)
            result["inference_throughput"] = round(throughput, 2)
            result["inference_peak_mem_gb"] = round(peak_mem, 2)
            result["inference_top_tokens"] = top_tokens[:4]
            result["inference_oom"] = False
            max_inference_bs = bs

            logger.info(f"    OK: {avg_time:.3f}s, {throughput:.1f} samples/s, peak={peak_mem:.1f}GB")
            logger.info(f"    Top tokens: {top_tokens[:4]}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                model.encode_images = original_fn
                torch.cuda.empty_cache()
                result["inference_oom"] = True
                logger.warning(f"    OOM at batch_size={bs} (inference)")
            else:
                model.encode_images = original_fn
                raise

        # --- Training test (forward + backward) ---
        logger.info(f"  [Training] batch_size={bs}...")
        try:
            # Left-pad full sequences (prompt + answer + eos)
            batch_full = all_full_seqs[:bs]
            batch_prompt_lens = all_prompt_lens[:bs]
            input_ids, attention_mask = _left_pad_sequences(batch_full, tokenizer.pad_token_id)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Build labels: -100 for left-pad and prompt tokens
            labels = input_ids.clone()
            for i in range(bs):
                seq_len = len(batch_full[i])
                pad_len = input_ids.shape[1] - seq_len
                labels[i, :pad_len + batch_prompt_lens[i]] = -100

            original_fn = model.encode_images
            model.encode_images = patched_encode

            # Warmup
            for _ in range(num_warmup):
                torch.cuda.reset_peak_memory_stats()
                output = model.forward(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    images=[torch.zeros(bs, 3, 384, 384, device=device, dtype=model.dtype) for _ in range(4)],
                    image_sizes=[(384, 384)] * bs,
                )
                output.loss.backward()
                torch.cuda.empty_cache()

            # Timed trials
            torch.cuda.synchronize()
            times = []
            losses = []
            for _ in range(num_trials):
                torch.cuda.reset_peak_memory_stats()
                for f in features:
                    if f.grad is not None:
                        f.grad.zero_()

                start = time.time()
                output = model.forward(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    images=[torch.zeros(bs, 3, 384, 384, device=device, dtype=model.dtype) for _ in range(4)],
                    image_sizes=[(384, 384)] * bs,
                )
                output.loss.backward()
                torch.cuda.synchronize()
                times.append(time.time() - start)
                losses.append(output.loss.item())

            model.encode_images = original_fn

            avg_time = sum(times) / len(times)
            avg_loss = sum(losses) / len(losses)
            peak_mem = max(torch.cuda.max_memory_allocated(i) for i in range(len(remapped_gpus))) / 1e9
            throughput = bs / avg_time

            # Check gradients flow to features
            grad_norms = [f.grad.norm().item() if f.grad is not None else 0.0 for f in features]

            result["training_time_s"] = round(avg_time, 3)
            result["training_throughput"] = round(throughput, 2)
            result["training_peak_mem_gb"] = round(peak_mem, 2)
            result["training_loss"] = round(avg_loss, 4)
            result["training_grad_norms"] = [round(g, 6) for g in grad_norms]
            result["training_oom"] = False
            max_training_bs = bs

            logger.info(f"    OK: {avg_time:.3f}s, {throughput:.1f} samples/s, peak={peak_mem:.1f}GB")
            logger.info(f"    Loss: {avg_loss:.4f}, grad norms: {[round(g, 4) for g in grad_norms]}")

            # Compare loss with baseline (should be similar for bs=1)
            if bs == 1 and baseline_losses:
                diff = abs(avg_loss - baseline_losses[0])
                logger.info(f"    Loss vs baseline: diff={diff:.4f} "
                            f"({'OK' if diff < 0.01 else 'MISMATCH!'})")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                model.encode_images = original_fn
                torch.cuda.empty_cache()
                result["training_oom"] = True
                logger.warning(f"    OOM at batch_size={bs} (training)")
            else:
                model.encode_images = original_fn
                raise

        results.append(result)
        torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("BATCHING BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"GPUs: {gpu_list}")
    print(f"Benchmark: {benchmark} ({len(samples)} questions)")
    print()

    # Inference table
    print("INFERENCE (forward only, no_grad):")
    print(f"{'BS':>4} {'Time(s)':>8} {'Samp/s':>8} {'Peak(GB)':>9} {'OOM':>5} {'Top tokens'}")
    print("-" * 70)
    for r in results:
        bs = r["batch_size"]
        if r.get("inference_oom"):
            print(f"{bs:4d}      --       --        --   OOM")
        else:
            print(f"{bs:4d} {r['inference_time_s']:8.3f} {r['inference_throughput']:8.1f} "
                  f"{r['inference_peak_mem_gb']:9.1f}    ok  {r.get('inference_top_tokens', [])}")

    print()
    print("TRAINING (forward + backward):")
    print(f"{'BS':>4} {'Time(s)':>8} {'Samp/s':>8} {'Peak(GB)':>9} {'Loss':>8} {'OOM':>5}")
    print("-" * 70)
    for r in results:
        bs = r["batch_size"]
        if r.get("training_oom"):
            print(f"{bs:4d}      --       --        --       --   OOM")
        else:
            print(f"{bs:4d} {r['training_time_s']:8.3f} {r['training_throughput']:8.1f} "
                  f"{r['training_peak_mem_gb']:9.1f} {r['training_loss']:8.4f}    ok")

    print()
    print(f"Max inference batch size: {max_inference_bs}")
    print(f"Max training batch size:  {max_training_bs}")

    if max_training_bs > 1:
        # Find the best throughput
        best = max((r for r in results if not r.get("training_oom")),
                   key=lambda r: r["training_throughput"])
        print(f"Best training throughput: {best['training_throughput']:.1f} samples/s at bs={best['batch_size']}")
        speedup = best["training_throughput"] / results[0]["training_throughput"]
        print(f"Speedup over bs=1: {speedup:.1f}x")
    elif max_training_bs == 1:
        print("Batching not viable for training — use single-sample gradient accumulation")
    else:
        print("ERROR: Even batch_size=1 failed!")


if __name__ == "__main__":
    fire.Fire(main)
