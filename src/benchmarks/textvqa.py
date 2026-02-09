"""TextVQA benchmark adapter — visual question answering requiring OCR."""

from typing import Iterator
from datasets import load_dataset
from PIL import Image

from .base import Benchmark, BenchmarkSample


def _vqa_accuracy(prediction: str, ground_truths: list[str]) -> float:
    """VQA accuracy: min(#humans who said prediction / 3, 1).

    Standard VQA evaluation metric with soft matching.
    """
    prediction = prediction.lower().strip()
    prediction = prediction.rstrip(".")

    count = sum(1 for gt in ground_truths if gt.lower().strip().rstrip(".") == prediction)
    return min(count / 3.0, 1.0)


class TextVQABenchmark(Benchmark):
    name = "textvqa"
    scoring_method = "vqa_accuracy"

    def load(self, max_samples: int | None = None) -> None:
        dataset = load_dataset("textvqa", split="validation")
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

            answers = row.get("answers", [])
            # Use most common answer as ground truth for simple scoring
            if answers:
                from collections import Counter
                ground_truth = Counter(answers).most_common(1)[0][0]
            else:
                ground_truth = ""

            yield BenchmarkSample(
                question_id=f"textvqa_{row.get('question_id', idx)}",
                question=row.get("question", ""),
                image=image.convert("RGB"),
                ground_truth=ground_truth,
                metadata={
                    "_index": idx,
                    "all_answers": answers,
                },
            )

    def format_question(self, sample: BenchmarkSample) -> str:
        return f"{sample.question}\nAnswer the question using a single word or phrase."

    def extract_answer(self, response: str, sample: BenchmarkSample) -> str:
        # Take the first line/sentence as the answer
        response = response.strip()
        response = response.split("\n")[0].strip()
        response = response.rstrip(".")
        return response

    def score(self, prediction: str, sample: BenchmarkSample) -> bool:
        # Use VQA accuracy against all annotator answers
        all_answers = sample.metadata.get("all_answers", [sample.ground_truth])
        return _vqa_accuracy(prediction, all_answers) >= 0.5
