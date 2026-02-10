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

    Args:
        model: The Cambrian model.
        features: List of 4 tensors (one per encoder) to inject.
    """
    original_fn = model.encode_images

    def patched_encode(image_aux_list, *args, **kwargs):
        return features

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
    if isinstance(image_tensor, list):
        image_tensor = [t.to(dtype=model.dtype, device=device) for t in image_tensor]
    else:
        image_tensor = [image_tensor.to(dtype=model.dtype, device=device)]

    with torch.no_grad():
        features = model.encode_images(image_tensor)

    shapes = [f.shape for f in features]
    logger.info(f"Encoder output shapes: {shapes}")
    return shapes
