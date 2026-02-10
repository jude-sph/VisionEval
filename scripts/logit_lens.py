"""Logit lens analysis of optimized noise embeddings.

Projects the optimized embeddings through the model's multimodal projector
and LM head to see what "words" the LLM reads from the noise.

Also captures intermediate hidden states at each transformer layer to show
how the model's interpretation evolves through the network.

Usage:
    python scripts/logit_lens.py
    python scripts/logit_lens.py --checkpoint results/optimization/tensors/mmmu_universal.pt
    python scripts/logit_lens.py --top_k 10
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
logger = logging.getLogger("logit_lens")

ENCODER_NAMES = ["SigLIP", "CLIP", "DINOv2", "ConvNeXt"]


def main(
    checkpoint: str = "results/optimization/tensors/mmmu_universal.pt",
    initial: str = "results/optimization/tensors/mmmu_universal_initial.pt",
    gpu_ids: str = "0,1,2,3",
    model_path: str = "nyu-visionx/cambrian-8b",
    top_k: int = 10,
    output_dir: str = "results/optimization/logit_lens",
):
    """Run logit lens analysis on saved embedding tensors.

    Args:
        checkpoint: Path to optimized embedding tensors (.pt file).
        initial: Path to initial (random) embedding tensors for comparison.
        gpu_ids: Comma-separated GPU indices.
        model_path: HuggingFace model path.
        top_k: Number of top tokens to report per position.
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
    logger.info("Model loaded")

    device = next(model.parameters()).device

    # Load embeddings
    logger.info(f"Loading optimized embeddings from {checkpoint}")
    optimized_tensors = torch.load(checkpoint, map_location=device, weights_only=True)
    optimized_tensors = [t.to(dtype=model.dtype) for t in optimized_tensors]

    has_initial = os.path.exists(initial)
    if has_initial:
        logger.info(f"Loading initial embeddings from {initial}")
        initial_tensors = torch.load(initial, map_location=device, weights_only=True)
        initial_tensors = [t.to(dtype=model.dtype) for t in initial_tensors]

    # --- Part 1: Project embeddings through mm_projector → lm_head ---
    logger.info("Running projector + lm_head analysis...")

    inner = getattr(model, "model", model)

    # Cambrian applies encode_images → then prepare_inputs_labels_for_multimodal
    # handles the projection internally. We need to replicate what happens
    # between encode_images output and the first transformer layer.
    #
    # The embeddings go through the model's forward path where they get
    # projected. To do logit lens at the input level, we run a minimal
    # forward pass and capture hidden states.

    from src.optimization.utils import encode_images_hook
    from cambrian.mm_utils import tokenizer_image_token
    from cambrian.constants import IMAGE_TOKEN_INDEX
    from src.model.inference import build_prompt

    # Create a minimal prompt with just image tokens
    prompt = build_prompt("Describe this image.", conv_mode="llama_3", include_image=True)
    input_ids = tokenizer_image_token(
        prompt=prompt,
        tokenizer=tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    # --- Register hooks to capture hidden states at every layer ---
    hidden_states_by_layer = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_states_by_layer[layer_idx] = h.detach().float().cpu()
        return hook_fn

    # Register hooks on each decoder layer
    hooks = []
    decoder_layers = getattr(inner, "layers", None) or getattr(inner, "model", None).layers
    logger.info(f"Found {len(decoder_layers)} decoder layers")
    for i, layer in enumerate(decoder_layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Run forward pass with optimized embeddings
    logger.info("Running forward pass with optimized embeddings...")
    with torch.no_grad(), encode_images_hook(model, optimized_tensors):
        output = model.forward(
            input_ids=input_ids,
            images=[
                torch.zeros(1, 3, 384, 384, device=device, dtype=model.dtype)
                for _ in range(4)
            ],
            image_sizes=[(384, 384)],
        )

    # Get the LM head and layer norm
    # Cambrian wraps LLaMA: model.lm_head, model.model.norm, model.model.layers
    lm_head = model.lm_head
    # Final layer norm (applied before lm_head in LLaMA)
    final_norm = getattr(inner, "norm", None) or getattr(inner, "final_layernorm", None)
    if final_norm is None:
        logger.warning("Could not find final layer norm, trying without it")

    # Find image token positions in the sequence
    # After prepare_inputs_labels_for_multimodal, image tokens are expanded
    # We'll analyze all positions but focus on the image token region
    total_seq_len = output.logits.shape[1]
    logger.info(f"Sequence length after image expansion: {total_seq_len}")

    # --- Project hidden states through lm_head at each layer ---
    logger.info("Projecting hidden states through lm_head at each layer...")
    results = {"layers": [], "metadata": {}}
    results["metadata"]["num_layers"] = len(decoder_layers)
    results["metadata"]["seq_len"] = total_seq_len
    results["metadata"]["checkpoint"] = checkpoint
    results["metadata"]["top_k"] = top_k

    for layer_idx in sorted(hidden_states_by_layer.keys()):
        h = hidden_states_by_layer[layer_idx].to(device).to(model.dtype)

        # Apply final layer norm then lm_head
        h_normed = final_norm(h) if final_norm is not None else h
        logits = lm_head(h_normed)  # [1, seq_len, vocab_size]

        # Get top-k tokens at each position
        probs = torch.softmax(logits[0].float(), dim=-1)
        top_probs, top_ids = probs.topk(top_k, dim=-1)

        # Sample positions: first 5 image tokens, middle, last 5, and text tokens
        # For a summary, aggregate across all image token positions
        layer_result = {
            "layer": layer_idx,
            "top_tokens_by_position": {},
        }

        # Aggregate: what are the most common top-1 tokens across all positions?
        top1_tokens = top_ids[:, 0].cpu().tolist()
        from collections import Counter
        token_counts = Counter(top1_tokens)
        most_common = token_counts.most_common(20)
        layer_result["most_common_top1"] = [
            {"token": tokenizer.decode([tok_id]), "token_id": tok_id, "count": count}
            for tok_id, count in most_common
        ]

        # Average entropy across positions
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        layer_result["mean_entropy"] = round(entropy.mean().item(), 4)

        # Top tokens at specific positions (first 3, middle, last 3)
        sample_positions = list(range(min(3, total_seq_len)))
        if total_seq_len > 6:
            sample_positions += [total_seq_len // 2]
            sample_positions += list(range(total_seq_len - 3, total_seq_len))

        for pos in sample_positions:
            tokens_at_pos = []
            for k in range(top_k):
                tok_id = top_ids[pos, k].item()
                tokens_at_pos.append({
                    "token": tokenizer.decode([tok_id]),
                    "token_id": tok_id,
                    "prob": round(top_probs[pos, k].item(), 6),
                })
            layer_result["top_tokens_by_position"][str(pos)] = tokens_at_pos

        results["layers"].append(layer_result)

        # Log summary for this layer
        top3_summary = ", ".join(
            f"'{tokenizer.decode([tok_id])}' ({count})"
            for tok_id, count in most_common[:5]
        )
        logger.info(
            f"  Layer {layer_idx:2d}: entropy={layer_result['mean_entropy']:.2f}  "
            f"top-1 tokens: {top3_summary}"
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # --- Part 2: Compare initial vs optimized at input level ---
    if has_initial:
        logger.info("Comparing initial vs optimized embeddings at input level...")

        # Run forward with initial embeddings too
        hidden_states_initial = {}

        hooks2 = []
        for i, layer in enumerate(decoder_layers):
            def make_hook_init(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    hidden_states_initial[layer_idx] = h.detach().float().cpu()
                return hook_fn
            h = layer.register_forward_hook(make_hook_init(i))
            hooks2.append(h)

        with torch.no_grad(), encode_images_hook(model, initial_tensors):
            model.forward(
                input_ids=input_ids,
                images=[
                    torch.zeros(1, 3, 384, 384, device=device, dtype=model.dtype)
                    for _ in range(4)
                ],
                image_sizes=[(384, 384)],
            )

        for h in hooks2:
            h.remove()

        # Compare hidden state norms and divergence per layer
        comparison = []
        for layer_idx in sorted(hidden_states_by_layer.keys()):
            h_opt = hidden_states_by_layer[layer_idx]
            h_init = hidden_states_initial.get(layer_idx)
            if h_init is not None:
                # Cosine similarity between initial and optimized hidden states
                cos_sim = torch.nn.functional.cosine_similarity(
                    h_opt.flatten(), h_init.flatten(), dim=0
                ).item()
                # L2 distance
                l2_dist = (h_opt - h_init).norm().item()
                comparison.append({
                    "layer": layer_idx,
                    "cosine_similarity": round(cos_sim, 6),
                    "l2_distance": round(l2_dist, 4),
                    "opt_norm": round(h_opt.norm().item(), 4),
                    "init_norm": round(h_init.norm().item(), 4),
                })

                if layer_idx % 8 == 0 or layer_idx == len(decoder_layers) - 1:
                    logger.info(
                        f"  Layer {layer_idx:2d}: cos_sim={cos_sim:.4f}  "
                        f"l2={l2_dist:.1f}  norm_opt={h_opt.norm():.1f}  norm_init={h_init.norm():.1f}"
                    )

        results["initial_vs_optimized"] = comparison

    # Save results
    output_path = os.path.join(output_dir, "logit_lens_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # --- Print readable summary ---
    print("\n" + "=" * 70)
    print("LOGIT LENS SUMMARY — Optimized Noise Embeddings")
    print("=" * 70)

    print("\nWhat the LLM 'reads' from the noise at each layer:")
    print(f"{'Layer':>6} {'Entropy':>8} {'Top tokens (by frequency across positions)'}")
    print("-" * 70)
    for layer_data in results["layers"]:
        top_tokens_str = "  ".join(
            f"{t['token']!r}({t['count']})"
            for t in layer_data["most_common_top1"][:6]
        )
        print(f"  {layer_data['layer']:4d}   {layer_data['mean_entropy']:7.2f}   {top_tokens_str}")

    if has_initial and "initial_vs_optimized" in results:
        print("\nDivergence from initial (random) embeddings:")
        print(f"{'Layer':>6} {'Cos Sim':>8} {'L2 Dist':>10}")
        print("-" * 30)
        for c in results["initial_vs_optimized"]:
            print(f"  {c['layer']:4d}   {c['cosine_similarity']:7.4f}   {c['l2_distance']:9.1f}")

    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
