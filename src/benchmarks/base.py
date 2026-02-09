"""Base class for benchmark adapters."""

from abc import ABC, abstractmethod
from PIL import Image
from typing import Any, Iterator
from dataclasses import dataclass


@dataclass
class BenchmarkSample:
    """A single benchmark question."""
    question_id: str
    question: str
    image: Image.Image
    ground_truth: str
    choices: list[str] | None = None  # For MCQ benchmarks
    metadata: dict[str, Any] | None = None


class Benchmark(ABC):
    """Base class for vision benchmark adapters."""

    name: str
    scoring_method: str  # "mc_accuracy", "binary_accuracy_f1", "vqa_accuracy", "exact_match"

    @abstractmethod
    def load(self, max_samples: int | None = None) -> None:
        """Load the dataset."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[BenchmarkSample]:
        ...

    @abstractmethod
    def format_question(self, sample: BenchmarkSample) -> str:
        """Format the question into a prompt string for the model.

        Should include answer format instructions (e.g., 'Answer with A, B, C, or D').
        Image tokens will be added by the inference wrapper.
        """
        ...

    @abstractmethod
    def extract_answer(self, response: str, sample: BenchmarkSample) -> str:
        """Extract the answer from model's raw response.

        For MCQ: extract the letter choice.
        For VQA: extract the short answer.
        """
        ...

    @abstractmethod
    def score(self, prediction: str, sample: BenchmarkSample) -> bool:
        """Score a single prediction against ground truth."""
        ...

    def get_raw_dataset(self):
        """Return the raw dataset for image swapping in wrong_image condition."""
        return getattr(self, "_dataset", None)
