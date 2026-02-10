"""Model loading wrapper for Cambrian-8B with multi-GPU and quantization support."""

import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


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
        # First GPU holds vision encoders (~3.8GB FP16) loaded separately by
        # Cambrian, so give it less headroom for LLM layers
        max_memory = {i: "11GiB" for i in gpu_ids}
        max_memory[gpu_ids[0]] = "7GiB"
        max_memory["cpu"] = "16GiB"
        kwargs["max_memory"] = max_memory
    elif gpu_ids is not None and len(gpu_ids) == 1 and not load_8bit:
        # Direct device placement — only works without quantization.
        # bitsandbytes quantized models can't be .to(device), so use "auto".
        kwargs["device_map"] = {"": gpu_ids[0]}
    else:
        kwargs["device_map"] = "auto"

    # Fall back gracefully if flash-attn not installed
    if use_flash_attn and not _flash_attn_available():
        logger.warning("flash-attn not installed, falling back to default attention")
        use_flash_attn = False

    # Pass device="cuda" so Cambrian's builder uses device_map instead of
    # calling model.to(device). This is required for bitsandbytes quantized
    # models which can't be .to(device), and for multi-GPU device_map.
    if gpu_ids is not None and len(gpu_ids) == 1 and not load_8bit:
        device = f"cuda:{gpu_ids[0]}"
    else:
        device = "cuda"

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
