"""Image conditions for VLM ablation evaluation."""

from .base import ImageCondition
from .normal import NormalCondition
from .no_image import NoImageCondition
from .wrong_image import WrongImageCondition

CONDITIONS = {
    "normal": NormalCondition,
    "no_image": NoImageCondition,
    "wrong_image": WrongImageCondition,
}


def get_condition(name: str) -> ImageCondition:
    """Get an image condition instance by name."""
    if name not in CONDITIONS:
        raise ValueError(f"Unknown condition: {name}. Available: {list(CONDITIONS.keys())}")
    return CONDITIONS[name]()
