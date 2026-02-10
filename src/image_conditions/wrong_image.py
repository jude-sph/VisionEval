"""Wrong-image condition: swap with a random different image from the dataset."""

import hashlib
import random
from PIL import Image
from typing import Optional, Any

from .base import ImageCondition


class WrongImageCondition(ImageCondition):
    name = "wrong_image"

    def apply(
        self,
        image: Image.Image,
        sample: dict[str, Any],
        dataset_images: Optional[Any] = None,
    ) -> Image.Image:
        if dataset_images is None:
            raise ValueError("WrongImageCondition requires dataset_images to swap from")

        # Deterministic seed from sample ID for reproducibility
        sample_id = str(sample.get("question_id", sample.get("id", "")))
        seed = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Pick a random different index
        current_idx = sample.get("_index")
        n = len(dataset_images)
        wrong_idx = rng.randint(0, n - 2)
        if current_idx is not None and wrong_idx >= current_idx:
            wrong_idx += 1

        # Store for metadata collection by the runner
        self.last_wrong_idx = wrong_idx

        # Extract image from the dataset row
        wrong_sample = dataset_images[wrong_idx]
        wrong_image = None
        if isinstance(wrong_sample, dict):
            # Try common image field names across benchmarks
            for key in ["image", "img", "image_1", "image_2", "image_3",
                         "image_4", "image_5", "image_6", "image_7"]:
                val = wrong_sample.get(key)
                if val is not None and isinstance(val, Image.Image):
                    wrong_image = val
                    break
        elif isinstance(wrong_sample, Image.Image):
            wrong_image = wrong_sample

        if isinstance(wrong_image, Image.Image):
            return wrong_image.convert("RGB")

        raise TypeError(f"Could not extract image from dataset sample at index {wrong_idx}")
