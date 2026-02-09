"""No-image condition: remove image entirely for text-only inference."""

from PIL import Image
from typing import Optional, Any

from .base import ImageCondition


class NoImageCondition(ImageCondition):
    name = "no_image"

    def apply(
        self,
        image: Image.Image,
        sample: dict[str, Any],
        dataset_images: Optional[Any] = None,
    ) -> None:
        return None
