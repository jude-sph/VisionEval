"""Differentiable reimplementation of Cambrian's process_images().

Replaces PIL-based preprocessing with pure PyTorch operations so gradients
can flow from the loss back through the vision encoders to the pixel tensor.

Each of Cambrian's 4 vision encoders (SigLIP, CLIP, DINOv2, ConvNeXt) has
its own image processor with a specific crop size and normalization params.
This module extracts those params once, then applies resize + normalize
using differentiable PyTorch ops (F.interpolate, elementwise math).
"""

import torch
import torch.nn.functional as F
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EncoderPreprocessParams:
    """Preprocessing parameters for one vision encoder."""
    crop_size: tuple[int, int]  # (height, width)
    mean: torch.Tensor  # [3] on target device
    std: torch.Tensor   # [3] on target device


def extract_preprocess_params(
    image_processor,
    device: torch.device,
) -> list[EncoderPreprocessParams]:
    """Extract preprocessing parameters from each encoder's image processor.

    Cambrian's image_processor is a list of 4 HuggingFace image processors,
    each with crop_size, image_mean, and image_std attributes.

    Args:
        image_processor: List of 4 per-encoder image processors.
        device: Device to place mean/std tensors on.

    Returns:
        List of 4 EncoderPreprocessParams.
    """
    if not isinstance(image_processor, (list, tuple)):
        image_processor = [image_processor]

    params = []
    for i, proc in enumerate(image_processor):
        # Extract crop size — could be dict {"height": H, "width": W} or int
        crop_size = getattr(proc, "crop_size", None) or getattr(proc, "size", None)
        if isinstance(crop_size, dict):
            h = crop_size.get("height", crop_size.get("shortest_edge", 384))
            w = crop_size.get("width", crop_size.get("shortest_edge", 384))
        elif isinstance(crop_size, int):
            h = w = crop_size
        else:
            h = w = 384  # fallback

        # Extract mean and std — lists of 3 floats (RGB)
        mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(proc, "image_std", [0.229, 0.224, 0.225])

        mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
        std_t = torch.tensor(std, dtype=torch.float32, device=device)

        params.append(EncoderPreprocessParams(
            crop_size=(h, w),
            mean=mean_t,
            std=std_t,
        ))
        logger.info(
            f"Encoder {i}: crop_size={h}x{w}, "
            f"mean={[round(m, 4) for m in mean]}, "
            f"std={[round(s, 4) for s in std]}"
        )

    return params


def differentiable_preprocess(
    pixels: torch.Tensor,
    params: list[EncoderPreprocessParams],
) -> list[torch.Tensor]:
    """Preprocess a pixel tensor for each encoder using differentiable ops.

    Replaces process_images() with:
      1. F.interpolate (bilinear) to each encoder's crop size
      2. Normalize with (x - mean) / std

    Args:
        pixels: Input pixel tensor [1, 3, H, W] in range [0, 1], float32.
        params: List of 4 EncoderPreprocessParams (from extract_preprocess_params).

    Returns:
        List of 4 tensors, each [1, 3, crop_h, crop_w] in float16,
        matching the format that model.forward(images=...) expects.
    """
    result = []
    for p in params:
        # Resize to encoder's expected input size
        resized = F.interpolate(
            pixels,
            size=p.crop_size,
            mode="bilinear",
            align_corners=False,
        )

        # Normalize: (x - mean) / std
        # mean/std are [3], reshape to [1, 3, 1, 1] for broadcasting
        mean = p.mean.view(1, 3, 1, 1)
        std = p.std.view(1, 3, 1, 1)
        normalized = (resized - mean) / std

        # Cast to float16 for the encoder (differentiable cast)
        result.append(normalized.to(torch.float16))

    return result
