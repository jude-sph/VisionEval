"""Model loading wrapper for Cambrian-8B with multi-GPU and quantization support."""

import torch
from typing import Optional


def load_cambrian(
    model_path: str = "nyu-visionx/cambrian-8b",
    model_name: str = "cambrian-8b",
    gpu_ids: Optional[list[int]] = None,
    load_8bit: bool = False,
    use_flash_attn: bool = True,
):
    """Load Cambrian model with appropriate device placement.

    Args:
        model_path: HuggingFace model path or local path.
        model_name: Model identifier for Cambrian's loader.
        gpu_ids: List of GPU indices. None = auto-detect.
            Single GPU: [0] -> single device.
            Multiple GPUs: [0, 1] -> tensor parallel via device_map.
        load_8bit: Use bitsandbytes INT8 quantization.
        use_flash_attn: Enable Flash Attention 2 for faster prefill.

    Returns:
        (tokenizer, model, image_processor, context_len)
    """
    from cambrian.model.builder import load_pretrained_model

    kwargs = {}

    # Device map for multi-GPU tensor parallelism
    if gpu_ids is not None and len(gpu_ids) > 1:
        # accelerate will split layers across specified GPUs
        kwargs["device_map"] = "auto"
        # Restrict to specified GPUs
        max_memory = {i: "11GiB" for i in gpu_ids}
        max_memory["cpu"] = "16GiB"
        kwargs["max_memory"] = max_memory
    elif gpu_ids is not None and len(gpu_ids) == 1:
        kwargs["device_map"] = {"": gpu_ids[0]}
    else:
        kwargs["device_map"] = "auto"

    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cuda"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=False,
        device=device,
        use_flash_attn=use_flash_attn,
        **kwargs,
    )

    model.eval()
    return tokenizer, model, image_processor, context_len
