"""Normal condition: pass-through baseline."""

from PIL import Image
from typing import Optional, Any

from .base import ImageCondition


class NormalCondition(ImageCondition):
    name = "normal"

    def apply(
        self,
        image: Image.Image,
        sample: dict[str, Any],
        dataset_images: Optional[Any] = None,
    ) -> Image.Image:
        return image
