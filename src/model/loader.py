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

    # Device map strategy:
    # - Multi-GPU: "auto" with max_memory constraints
    # - Single GPU FP16: {"": device_id} for direct placement
    # - Single GPU INT8: "auto" + device="cuda" to avoid Cambrian's builder
    #   overriding device_map (it sets {"": device} when device != "cuda",
    #   which triggers accelerate's .to() crash for bitsandbytes models).
    #   After loading, we fix any CPU-stranded buffers manually.
    if gpu_ids is not None and len(gpu_ids) > 1:
        kwargs["device_map"] = "auto"
        max_memory = {i: "11GiB" for i in gpu_ids}
        max_memory[gpu_ids[0]] = "7GiB"
        max_memory["cpu"] = "16GiB"
        kwargs["max_memory"] = max_memory
    elif gpu_ids is not None and len(gpu_ids) == 1 and not load_8bit:
        kwargs["device_map"] = {"": gpu_ids[0]}
    else:
        kwargs["device_map"] = "auto"

    # Fall back gracefully if flash-attn not installed
    if use_flash_attn and not _flash_attn_available():
        logger.warning("flash-attn not installed, falling back to default attention")
        use_flash_attn = False

    # For INT8: must use device="cuda" so Cambrian's builder doesn't override
    # device_map to {"": device} which crashes with bitsandbytes.
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

    # Fix INT8 device placement: device_map="auto" can leave some buffers
    # (e.g., rotary embeddings inv_freq) on CPU. Move them to the target GPU.
    if load_8bit:
        target_device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cuda:0")
        moved = 0
        for name, buf in model.named_buffers():
            if buf.device.type == "cpu":
                buf.data = buf.data.to(target_device)
                moved += 1
        for name, param in model.named_parameters():
            if param.device.type == "cpu" and not hasattr(param, "quant_state"):
                param.data = param.data.to(target_device)
                moved += 1
        if moved > 0:
            logger.info(f"Moved {moved} CPU tensors to {target_device} after INT8 loading")

    model.eval()
    return tokenizer, model, image_processor, context_len
