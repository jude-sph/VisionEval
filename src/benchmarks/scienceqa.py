"""ScienceQA-Image benchmark adapter — science questions with diagrams."""

import re
from typing import Iterator
from datasets import load_dataset
from PIL import Image

from .base import Benchmark, BenchmarkSample


class ScienceQABenchmark(Benchmark):
    name = "scienceqa"
    scoring_method = "mc_accuracy"

    def load(self, max_samples: int | None = None) -> None:
        dataset = load_dataset("derek-thomas/ScienceQA", split="test")
        # Filter to only questions with images
        dataset = dataset.filter(lambda x: x.get("image") is not None)
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

            choices_list = row.get("choices", [])
            choices = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(choices_list)]
            answer_idx = row.get("answer", 0)
            ground_truth = chr(65 + answer_idx)  # Convert index to letter

            context = row.get("hint", "")
            question_text = row.get("question", "")
            if context:
                question_text = f"Context: {context}\nQuestion: {question_text}"

            yield BenchmarkSample(
                question_id=f"sqa_{idx}",
                question=question_text,
                image=image.convert("RGB"),
                ground_truth=ground_truth,
                choices=choices,
                metadata={"_index": idx, "subject": row.get("subject", "")},
            )

    def format_question(self, sample: BenchmarkSample) -> str:
        choices_str = "\n".join(sample.choices) if sample.choices else ""
        return (
            f"{sample.question}\n{choices_str}\n"
            f"Answer with the option letter (e.g., A, B, C, or D)."
        )

    def extract_answer(self, response: str, sample: BenchmarkSample) -> str:
        response = response.strip()
        n_choices = len(sample.choices) if sample.choices else 4
        valid_letters = [chr(65 + i) for i in range(n_choices)]
        pattern = r"\b([" + "".join(valid_letters) + r"])\b"
        match = re.search(pattern, response.upper())
        if match:
            return match.group(1)
        if response and response[0].upper() in valid_letters:
            return response[0].upper()
        return response[:1].upper() if response else ""

    def score(self, prediction: str, sample: BenchmarkSample) -> bool:
        return prediction.upper().strip() == sample.ground_truth.upper().strip()
