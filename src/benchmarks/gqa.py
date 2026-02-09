"""GQA benchmark adapter — compositional visual reasoning from scene graphs."""

from typing import Iterator
from datasets import load_dataset
from PIL import Image

from .base import Benchmark, BenchmarkSample


class GQABenchmark(Benchmark):
    name = "gqa"
    scoring_method = "exact_match"

    def load(self, max_samples: int | None = None) -> None:
        dataset = load_dataset("lmms-lab/GQA", "testdev_balanced", split="testdev")
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
                question_id=f"gqa_{row.get('question_id', idx)}",
                question=row.get("question", ""),
                image=image.convert("RGB"),
                ground_truth=row.get("answer", "").lower().strip(),
                metadata={"_index": idx},
            )

    def format_question(self, sample: BenchmarkSample) -> str:
        return f"{sample.question}\nAnswer the question using a single word or phrase."

    def extract_answer(self, response: str, sample: BenchmarkSample) -> str:
        response = response.strip().lower()
        response = response.split("\n")[0].strip()
        response = response.rstrip(".")
        return response

    def score(self, prediction: str, sample: BenchmarkSample) -> bool:
        return prediction.lower().strip() == sample.ground_truth.lower().strip()
