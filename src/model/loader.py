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

    if gpu_ids is not None and len(gpu_ids) == 1 and not load_8bit:
        device = f"cuda:{gpu_ids[0]}"
    else:
        device = "cuda"

    # Patch accelerate's dispatch_model to handle bitsandbytes quantized models.
    # Older versions of accelerate call model.to(device) inside dispatch_model,
    # which bitsandbytes forbids. The model is already on the correct device
    # after quantized loading, so we can safely skip the .to() call.
    if load_8bit:
        import accelerate.big_modeling as _abm
        _original_dispatch = _abm.dispatch_model

        def _safe_dispatch(model, device_map, **dm_kwargs):
            try:
                return _original_dispatch(model, device_map, **dm_kwargs)
            except ValueError as e:
                if "not supported for" in str(e) and "bitsandbytes" in str(e):
                    logger.info("Skipped dispatch_model .to() for quantized model")
                    return model
                raise

        _abm.dispatch_model = _safe_dispatch

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

    # Restore original dispatch_model
    if load_8bit:
        _abm.dispatch_model = _original_dispatch

    model.eval()
    return tokenizer, model, image_processor, context_len
