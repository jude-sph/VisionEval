"""MMBench benchmark adapter — multilingual multiple-choice VQA."""

import re
from typing import Iterator
from datasets import load_dataset
from PIL import Image

from .base import Benchmark, BenchmarkSample


class MMBenchBenchmark(Benchmark):
    name = "mmbench"
    scoring_method = "mc_accuracy"

    def load(self, max_samples: int | None = None) -> None:
        dataset = load_dataset("opencompass/MMBench", split="dev")
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

            # Build choices from A, B, C, D columns
            choices = []
            for letter in ["A", "B", "C", "D"]:
                opt = row.get(letter, "")
                if opt:
                    choices.append(f"{letter}. {opt}")

            yield BenchmarkSample(
                question_id=f"mmbench_{row.get('index', idx)}",
                question=row.get("question", ""),
                image=image.convert("RGB"),
                ground_truth=row.get("answer", ""),
                choices=choices,
                metadata={"_index": idx, "category": row.get("category", "")},
            )

    def format_question(self, sample: BenchmarkSample) -> str:
        choices_str = "\n".join(sample.choices) if sample.choices else ""
        return (
            f"{sample.question}\n{choices_str}\n"
            f"Answer with the option letter (e.g., A, B, C, or D)."
        )

    def extract_answer(self, response: str, sample: BenchmarkSample) -> str:
        response = response.strip()
        match = re.search(r"\b([A-D])\b", response.upper())
        if match:
            return match.group(1)
        if response and response[0].upper() in "ABCD":
            return response[0].upper()
        return response[:1].upper() if response else ""

    def score(self, prediction: str, sample: BenchmarkSample) -> bool:
        return prediction.upper().strip() == sample.ground_truth.upper().strip()
