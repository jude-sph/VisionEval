"""Utilities for noise optimization: context managers and helpers."""

import torch
import logging
from contextlib import contextmanager
from PIL import Image

from cambrian.mm_utils import process_images

logger = logging.getLogger(__name__)


@contextmanager
def encode_images_hook(model, features: list[torch.Tensor]):
    """Monkey-patch model.encode_images to return pre-computed features.

    This bypasses the vision encoders entirely, allowing us to optimize
    the encoder output tensors directly via gradient descent.

    Features may be float32 (for optimizer stability) — they are cast to
    the model's dtype (float16) inside the hook. The cast is differentiable,
    so gradients flow back to the float32 tensors.

    Args:
        model: The Cambrian model.
        features: List of 4 tensors (one per encoder) to inject.
    """
    original_fn = model.encode_images
    target_dtype = model.dtype

    def patched_encode(image_aux_list, *args, **kwargs):
        return [f.to(target_dtype) for f in features]

    model.encode_images = patched_encode
    try:
        yield
    finally:
        model.encode_images = original_fn


def get_encoder_output_shapes(
    model,
    image_processor,
    image_size: tuple[int, int] = (384, 384),
) -> list[tuple[int, ...]]:
    """Discover the output shapes of each vision encoder.

    Runs a dummy image through the encoders to determine shapes.

    Args:
        model: Loaded Cambrian model.
        image_processor: List of per-encoder image processors.
        image_size: Size of dummy image to use.

    Returns:
        List of 4 shape tuples, e.g. [(1, 576, 1152), (1, 576, 1024), ...].
    """
    dummy_image = Image.new("RGB", image_size, color=(128, 128, 128))
    device = next(model.parameters()).device

    image_tensor = process_images(
        images=[dummy_image],
        image_processor=image_processor,
        model_cfg=model.config,
    )
    # Always use float16 — encoder inputs expect float16 regardless of INT8 quantization
    if isinstance(image_tensor, list):
        image_tensor = [t.to(dtype=torch.float16, device=device) for t in image_tensor]
    else:
        image_tensor = [image_tensor.to(dtype=torch.float16, device=device)]

    with torch.no_grad():
        features = model.encode_images(image_tensor)

    shapes = [f.shape for f in features]
    logger.info(f"Encoder output shapes: {shapes}")
    return shapes


@contextmanager
def enable_vision_grad(model, exclude_indices: list[int] | None = None):
    """Enable gradient flow through vision encoders.

    Cambrian's vision towers wrap their forward pass in
    torch.set_grad_enabled(self.unfreeze_mm_vision_tower).
    This context manager sets that flag to True on selected towers,
    then restores the original values on exit.

    Args:
        model: The Cambrian model.
        exclude_indices: Encoder indices to skip (leave frozen).
            E.g. [3] to exclude ConvNeXt, which needs too much memory
            for backward on 12GB GPUs (1024x1024 input, 9216 tokens).
    """
    inner = getattr(model, "model", model)
    towers = getattr(inner, "vision_tower_aux_list", []) or []
    exclude = set(exclude_indices or [])

    # Save original values
    original_flags = []
    for i, tower in enumerate(towers):
        original_flags.append(getattr(tower, "unfreeze_mm_vision_tower", False))
        if i not in exclude:
            tower.unfreeze_mm_vision_tower = True

    try:
        yield
    finally:
        for tower, flag in zip(towers, original_flags):
            tower.unfreeze_mm_vision_tower = flag
