"""POPE benchmark adapter — object hallucination probing (binary yes/no)."""

import re
from typing import Iterator
from datasets import load_dataset
from PIL import Image

from .base import Benchmark, BenchmarkSample


class POPEBenchmark(Benchmark):
    name = "pope"
    scoring_method = "binary_accuracy_f1"

    def load(self, max_samples: int | None = None) -> None:
        dataset = load_dataset("lmms-lab/POPE", split="test")
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        for idx, row in enumerate(self._dataset):
            image = row.get("image")
            if image is None:
                continue
            if not isinstance(image, Image.Image):
                continue

            yield BenchmarkSample(
                question_id=f"pope_{idx}",
                question=row.get("question", ""),
                image=image.convert("RGB"),
                ground_truth=row.get("answer", "").lower().strip(),
                metadata={
                    "_index": idx,
                    "category": row.get("category", ""),
                },
            )

    def format_question(self, sample: BenchmarkSample) -> str:
        return f"{sample.question}\nAnswer with yes or no."

    def extract_answer(self, response: str, sample: BenchmarkSample) -> str:
        response = response.lower().strip()
        # Look for yes/no
        if re.search(r"\byes\b", response):
            return "yes"
        if re.search(r"\bno\b", response):
            return "no"
        # Fallback: first word
        first_word = response.split()[0] if response.split() else ""
        return first_word

    def score(self, prediction: str, sample: BenchmarkSample) -> bool:
        return prediction.lower().strip() == sample.ground_truth.lower().strip()
