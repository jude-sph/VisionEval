"""Model loading wrapper for Cambrian-8B with multi-GPU support."""

import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    # Flash Attention 2 requires compute capability >= 8.0 (Ampere+).
    # It may be installed but will crash on Pascal (6.1) or older GPUs.
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            logger.info(
                f"flash-attn installed but GPU compute {cap[0]}.{cap[1]} < 8.0; disabling"
            )
            return False
    return True


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
        load_8bit: Ignored (INT8 not supported with this Cambrian/accelerate
            version). Always loads in FP16.
        use_flash_attn: Enable Flash Attention 2 for faster prefill.

    Returns:
        (tokenizer, model, image_processor, context_len)
    """
    if load_8bit:
        logger.warning(
            "load_8bit=True requested but INT8 is not compatible with this "
            "Cambrian/accelerate version. Loading in FP16 instead."
        )

    from cambrian.model.builder import load_pretrained_model

    kwargs = {}

    if gpu_ids is not None and len(gpu_ids) > 1:
        kwargs["device_map"] = "auto"
        max_memory = {i: "11GiB" for i in gpu_ids}
        max_memory[gpu_ids[0]] = "7GiB"
        max_memory["cpu"] = "16GiB"
        kwargs["max_memory"] = max_memory
    elif gpu_ids is not None and len(gpu_ids) == 1:
        kwargs["device_map"] = {"": gpu_ids[0]}
    else:
        kwargs["device_map"] = "auto"

    if use_flash_attn and not _flash_attn_available():
        logger.warning("flash-attn not installed, falling back to default attention")
        use_flash_attn = False

    if gpu_ids is not None and len(gpu_ids) == 1:
        device = f"cuda:{gpu_ids[0]}"
    else:
        device = "cuda"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device=device,
        use_flash_attn=use_flash_attn,
        **kwargs,
    )

    model.eval()
    return tokenizer, model, image_processor, context_len
