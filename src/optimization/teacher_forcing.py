"""Teacher-forcing loss: differentiable loss from model's forward pass."""

import torch
import logging

from cambrian.mm_utils import tokenizer_image_token
from cambrian.constants import IMAGE_TOKEN_INDEX

from src.model.inference import build_prompt

logger = logging.getLogger(__name__)


def compute_teacher_forcing_loss(
    model,
    tokenizer,
    question: str,
    answer: str,
    image_sizes: list[tuple[int, int]] | None = None,
    conv_mode: str = "llama_3",
) -> torch.Tensor:
    """Compute cross-entropy loss for the correct answer via teacher forcing.

    The model's encode_images should be patched (via encode_images_hook)
    before calling this function, so it returns the optimizable features.

    We build the full prompt + answer, create labels that mask the prompt
    tokens (-100), and call model.forward() to get the loss on answer tokens.

    Args:
        model: Cambrian model (with encode_images already patched).
        tokenizer: Tokenizer.
        question: Formatted question string.
        answer: Ground truth answer text (e.g., "A", "yes", "blue sky").
        image_sizes: List of (width, height) tuples for the images.
        conv_mode: Conversation template name.

    Returns:
        Scalar loss tensor (differentiable).
    """
    device = next(model.parameters()).device

    # Build prompt with image tokens included
    prompt = build_prompt(question, conv_mode=conv_mode, include_image=True)

    # Tokenize prompt (with IMAGE_TOKEN_INDEX placeholders)
    prompt_ids = tokenizer_image_token(
        prompt=prompt,
        tokenizer=tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )  # shape: [seq_len]

    # Tokenize the answer
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_ids = torch.tensor(answer_ids, dtype=torch.long)

    # Concatenate: prompt + answer + EOS
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        full_ids = torch.cat([prompt_ids, answer_ids, torch.tensor([eos_id])])
    else:
        full_ids = torch.cat([prompt_ids, answer_ids])

    full_ids = full_ids.unsqueeze(0).to(device)  # [1, seq_len]

    # Build labels: -100 for prompt tokens (ignored in loss), real IDs for answer
    labels = full_ids.clone()
    prompt_len = len(prompt_ids)
    labels[0, :prompt_len] = -100  # mask prompt tokens

    # We need dummy image inputs to trigger prepare_inputs_labels_for_multimodal.
    # The actual features come from the patched encode_images (via encode_images_hook).
    # Format: list of per-encoder tensors, matching what process_images() returns.
    # Content doesn't matter since encode_images is patched.
    num_encoders = 4  # SigLIP, CLIP, DINOv2, ConvNeXt
    dummy_images = [
        torch.zeros(1, 3, 384, 384, device=device, dtype=model.dtype)
        for _ in range(num_encoders)
    ]
    if image_sizes is None:
        image_sizes = [(384, 384)]

    # Forward pass — model.forward() calls prepare_inputs_labels_for_multimodal
    # which calls encode_images (our patched version), then computes loss.
    output = model.forward(
        input_ids=full_ids,
        labels=labels,
        images=dummy_images,
        image_sizes=image_sizes,
    )

    return output.loss
