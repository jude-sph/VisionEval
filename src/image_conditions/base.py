"""Base class for image conditions applied before model inference."""

from abc import ABC, abstractmethod
from PIL import Image
from typing import Optional, Any


class ImageCondition(ABC):
    """Transform applied to an image before it's processed by the model."""

    name: str

    @abstractmethod
    def apply(
        self,
        image: Image.Image,
        sample: dict[str, Any],
        dataset_images: Optional[Any] = None,
    ) -> Optional[Image.Image]:
        """Transform the image.

        Args:
            image: Original PIL Image (RGB).
            sample: Full data sample dict (question_id, question, etc.).
            dataset_images: Reference to the full dataset for swap-based conditions.

        Returns:
            Transformed PIL Image, or None for no-image condition.
        """
        ...
