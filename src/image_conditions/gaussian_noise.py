"""Gaussian noise condition: replace image with random RGB noise at the same resolution."""

import hashlib
import numpy as np
from PIL import Image
from typing import Optional, Any

from .base import ImageCondition


class GaussianNoiseCondition(ImageCondition):
    name = "gaussian_noise"

    def apply(
        self,
        image: Image.Image,
        sample: dict[str, Any],
        dataset_images: Optional[Any] = None,
    ) -> Image.Image:
        # Deterministic seed from sample ID for reproducibility
        sample_id = str(sample.get("question_id", sample.get("id", "")))
        seed = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)

        # Generate random RGB noise at the same resolution as the original
        w, h = image.size
        noise = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(noise, "RGB")
