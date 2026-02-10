"""Inference wrapper for Cambrian model: handles both image and no-image paths."""

import math
import re
import torch
from PIL import Image
from typing import Optional

from cambrian.mm_utils import process_images, tokenizer_image_token
from cambrian.conversation import conv_templates
from cambrian.constants import IMAGE_TOKEN_INDEX


def _clean_dataset_image_refs(text: str) -> str:
    """Strip dataset-level image references like <image 1>, <image 2>, etc."""
    return re.sub(r"<image\s+\d+>", "", text)


def build_prompt(question: str, conv_mode: str = "llama_3", include_image: bool = True) -> str:
    """Build a Cambrian-formatted prompt from a question string.

    Args:
        question: The question text (may already include image tokens).
        conv_mode: Conversation template name.
        include_image: If False, strip image tokens from the prompt.

    Returns:
        Formatted prompt string.
    """
    conv = conv_templates[conv_mode].copy()

    # Strip dataset-level image placeholders (e.g. MMMU's <image 1>)
    question = _clean_dataset_image_refs(question)

    # Ensure image tokens are present if needed
    if include_image and "<image>" not in question:
        question = f"<im_start><image><im_end>\n{question}"
    elif not include_image:
        # Strip all image-related tokens
        for token in ["<im_start>", "<image>", "<im_end>", "<image-placeholder>"]:
            question = question.replace(token, "")
        question = question.strip()

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_inference(
    model,
    tokenizer,
    image_processor,
    question: str,
    image: Optional[Image.Image] = None,
    conv_mode: str = "llama_3",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> dict:
    """Run single-sample inference on the Cambrian model.

    Args:
        model: Loaded Cambrian model.
        tokenizer: Associated tokenizer.
        image_processor: Image preprocessing pipeline.
        question: Question text (may include image tokens).
        image: PIL Image or None for text-only inference.
        conv_mode: Conversation template name.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        Dict with keys:
            response: Generated answer string.
            first_token_probs: Dict mapping answer options to {prob, logprob}.
            response_logprob: Total log-probability of the response.
            response_perplexity: Perplexity of the response.
            num_generated_tokens: Number of tokens generated.
    """
    include_image = image is not None
    prompt = build_prompt(question, conv_mode=conv_mode, include_image=include_image)

    # Tokenize prompt
    input_ids = tokenizer_image_token(
        prompt=prompt,
        tokenizer=tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)

    # Move input_ids to model device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Process image if provided
    image_tensor = None
    image_sizes = None
    if image is not None:
        image_tensor = process_images(
            images=[image],
            image_processor=image_processor,
            model_cfg=model.config,
        )
        if isinstance(image_tensor, list):
            image_tensor = [t.to(dtype=model.dtype, device=device) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=model.dtype, device=device)
        image_sizes = [image.size]

    # Generate with score output for logprob extraction
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    output_ids = outputs.sequences
    scores = outputs.scores  # tuple of [batch, vocab_size] logit tensors

    # Decode — Cambrian's generate() may return only new tokens (when it
    # passes inputs_embeds internally) or full sequence (input + generated).
    if output_ids.shape[1] > input_ids.shape[1]:
        generated_ids = output_ids[:, input_ids.shape[1]:]
    else:
        generated_ids = output_ids
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # --- Logprob extraction ---
    # Per-token log-probabilities for the generated sequence
    token_logprobs = []
    if scores and generated_ids.shape[1] > 0:
        for i, score in enumerate(scores):
            if i >= generated_ids.shape[1]:
                break
            log_probs = torch.log_softmax(score[0], dim=-1)
            token_id = generated_ids[0, i].item()
            token_logprobs.append(log_probs[token_id].item())

    num_tokens = len(token_logprobs)
    response_logprob = sum(token_logprobs) if token_logprobs else None
    response_perplexity = None
    if num_tokens > 0 and response_logprob is not None:
        response_perplexity = math.exp(-response_logprob / num_tokens)

    # First-token probabilities for common answer options.
    # For each canonical key (A/B/C/D/yes/no), sum probabilities across
    # tokenizer variants (e.g. "yes" and "Yes" may be different tokens).
    first_token_probs = {}
    if scores:
        first_logits = scores[0][0]  # [vocab_size]
        first_probs = torch.softmax(first_logits, dim=-1)

        answer_variants = {
            "A": ["A"], "B": ["B"], "C": ["C"], "D": ["D"],
            "yes": ["yes", "Yes"], "no": ["no", "No"],
        }
        for key, variants in answer_variants.items():
            tids = set()
            for v in variants:
                ids = tokenizer.encode(v, add_special_tokens=False)
                if ids:
                    tids.add(ids[0])
            if tids:
                prob = sum(first_probs[tid].item() for tid in tids)
                logprob = math.log(prob) if prob > 0 else float("-inf")
                first_token_probs[key] = {
                    "prob": round(prob, 6),
                    "logprob": round(logprob, 4),
                }

    return {
        "response": response,
        "first_token_probs": first_token_probs,
        "response_logprob": round(response_logprob, 4) if response_logprob is not None else None,
        "response_perplexity": round(response_perplexity, 4) if response_perplexity is not None else None,
        "num_generated_tokens": num_tokens,
        "prompt_token_count": input_ids.shape[1],
    }
