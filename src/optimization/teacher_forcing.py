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
    images: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute cross-entropy loss for the correct answer via teacher forcing.

    For embedding-space optimization: encode_images should be patched (via
    encode_images_hook) before calling, and images can be left as None.

    For pixel-space optimization: pass preprocessed image tensors via the
    images parameter (list of 4 tensors from differentiable_preprocess).

    We build the full prompt + answer, create labels that mask the prompt
    tokens (-100), and call model.forward() to get the loss on answer tokens.

    Args:
        model: Cambrian model.
        tokenizer: Tokenizer.
        question: Formatted question string.
        answer: Ground truth answer text (e.g., "A", "yes", "blue sky").
        image_sizes: List of (width, height) tuples for the images.
        conv_mode: Conversation template name.
        images: Optional list of preprocessed image tensors (for pixel-space
            optimization). If None, uses dummy images (for embedding-space
            optimization where encode_images is patched).

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

    # Image tensors for the forward pass.
    # For pixel-space optimization: use the provided preprocessed tensors.
    # For embedding-space optimization: use dummy tensors (encode_images is patched).
    if images is not None:
        forward_images = images
    else:
        num_encoders = 4  # SigLIP, CLIP, DINOv2, ConvNeXt
        forward_images = [
            torch.zeros(1, 3, 384, 384, device=device, dtype=model.dtype)
            for _ in range(num_encoders)
        ]
    if image_sizes is None:
        image_sizes = [(384, 384)]

    # Forward pass — model.forward() calls prepare_inputs_labels_for_multimodal
    # which calls encode_images, then computes loss.
    output = model.forward(
        input_ids=full_ids,
        labels=labels,
        images=forward_images,
        image_sizes=image_sizes,
    )

    return output.loss


def compute_batched_teacher_forcing_loss(
    model,
    tokenizer,
    questions: list[str],
    answers: list[str],
    conv_mode: str = "llama_3",
) -> torch.Tensor:
    """Compute teacher-forcing loss for a batch of questions.

    Left-pads all sequences so that position -1 is always the last real token.
    Labels use -100 for pad + prompt tokens, real IDs for answer tokens only.

    encode_images must be patched (via encode_images_hook_batched) before calling.

    Args:
        model: Cambrian model.
        tokenizer: Tokenizer.
        questions: List of formatted question strings.
        answers: List of ground truth answer strings.
        conv_mode: Conversation template name.

    Returns:
        Scalar loss tensor (differentiable), averaged across batch.
    """
    device = next(model.parameters()).device
    batch_size = len(questions)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # Tokenize each question + answer
    all_full_ids = []
    all_prompt_lens = []

    for question, answer in zip(questions, answers):
        prompt = build_prompt(question, conv_mode=conv_mode, include_image=True)
        prompt_ids = tokenizer_image_token(
            prompt=prompt,
            tokenizer=tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )  # [seq_len]

        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        answer_ids = torch.tensor(answer_ids, dtype=torch.long)

        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            full_ids = torch.cat([prompt_ids, answer_ids, torch.tensor([eos_id])])
        else:
            full_ids = torch.cat([prompt_ids, answer_ids])

        all_full_ids.append(full_ids.tolist())
        all_prompt_lens.append(len(prompt_ids))

    # Left-pad to max length
    max_len = max(len(seq) for seq in all_full_ids)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, (seq, prompt_len) in enumerate(zip(all_full_ids, all_prompt_lens)):
        seq_len = len(seq)
        pad_len = max_len - seq_len
        input_ids[i, pad_len:] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, pad_len:] = 1
        # Labels: -100 for pad + prompt, real IDs for answer + eos
        labels[i, pad_len + prompt_len:] = torch.tensor(seq[prompt_len:], dtype=torch.long)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # Dummy images (encode_images is patched to return learned features)
    forward_images = [
        torch.zeros(batch_size, 3, 384, 384, device=device, dtype=model.dtype)
        for _ in range(4)
    ]

    output = model.forward(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=forward_images,
        image_sizes=[(384, 384)] * batch_size,
    )

    return output.loss


def check_answers_batched(
    model,
    tokenizer,
    features: list[torch.Tensor],
    samples,
    benchmark,
    device: torch.device,
    conv_mode: str = "llama_3",
    batch_size: int = 4,
) -> int:
    """Evaluate accuracy on a list of samples using batched inference.

    Left-pads prompt-only sequences. logits[:, -1].argmax() gives next token
    for all batch items since left-padding ensures the last position is the
    last real token.

    Args:
        model: Cambrian model.
        tokenizer: Tokenizer.
        features: List of 4 learnable feature tensors.
        samples: List of BenchmarkSample objects.
        benchmark: Benchmark instance (for extract_answer, score).
        device: CUDA device.
        conv_mode: Conversation template name.
        batch_size: Number of samples per forward pass.

    Returns:
        Number of correct predictions.
    """
    from src.optimization.utils import encode_images_hook_batched

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    correct = 0

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start + batch_size]
        bs = len(batch)

        # Tokenize prompt-only for each sample
        all_seqs = []
        for sample in batch:
            question_text = benchmark.format_question(sample)
            prompt = build_prompt(question_text, conv_mode=conv_mode, include_image=True)
            prompt_ids = tokenizer_image_token(
                prompt=prompt,
                tokenizer=tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).tolist()
            all_seqs.append(prompt_ids)

        # Left-pad
        max_len = max(len(s) for s in all_seqs)
        input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
        for i, seq in enumerate(all_seqs):
            pad_len = max_len - len(seq)
            input_ids[i, pad_len:] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, pad_len:] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        forward_images = [
            torch.zeros(bs, 3, 384, 384, device=device, dtype=model.dtype)
            for _ in range(4)
        ]

        with torch.no_grad(), encode_images_hook_batched(model, features, bs):
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=forward_images,
                image_sizes=[(384, 384)] * bs,
            )

        # logits[:, -1] is the next-token prediction for each item
        next_token_ids = output.logits[:, -1].argmax(dim=-1)  # [bs]

        for i, sample in enumerate(batch):
            response = tokenizer.decode([next_token_ids[i].item()]).strip()
            prediction = benchmark.extract_answer(response, sample)
            if benchmark.score(prediction, sample):
                correct += 1

    return correct
