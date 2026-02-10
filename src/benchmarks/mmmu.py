"""MMMU benchmark adapter — college-level multimodal exam questions."""

import re
from typing import Iterator
from datasets import load_dataset, concatenate_datasets
from PIL import Image

from .base import Benchmark, BenchmarkSample


class MMMUBenchmark(Benchmark):
    name = "mmmu"
    scoring_method = "mc_accuracy"

    def load(self, max_samples: int | None = None) -> None:
        # MMMU validation set spans multiple subjects — must load each and concatenate
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names("MMMU/MMMU")
        splits = []
        for config in configs:
            splits.append(load_dataset("MMMU/MMMU", config, split="validation"))
        dataset = concatenate_datasets(splits)
        # Filter to rows with at least one image before limiting,
        # so max_samples controls actual yielded samples
        img_keys = ["image_1", "image_2", "image_3", "image_4",
                     "image_5", "image_6", "image_7"]
        dataset = dataset.filter(
            lambda row: any(row.get(k) is not None for k in img_keys)
        )
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        for idx, row in enumerate(self._dataset):
            # MMMU stores images as image_1, image_2, etc.
            image = None
            for img_key in ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]:
                if row.get(img_key) is not None:
                    image = row[img_key]
                    break

            if image is None:
                continue

            if not isinstance(image, Image.Image):
                continue

            # Build choices from options field
            choices = []
            options = row.get("options", [])
            if isinstance(options, str):
                # Parse string format like "['A', 'B', 'C', 'D']"
                options = eval(options) if options.startswith("[") else options.split("\n")
            for i, opt in enumerate(options):
                choices.append(f"{chr(65 + i)}. {opt}")

            yield BenchmarkSample(
                question_id=f"mmmu_{row.get('id', idx)}",
                question=row["question"],
                image=image.convert("RGB"),
                ground_truth=row["answer"],
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
        # Try to find a single letter answer
        match = re.search(r"\b([A-D])\b", response.upper())
        if match:
            return match.group(1)
        # If response starts with a letter
        if response and response[0].upper() in "ABCD":
            return response[0].upper()
        return response[:1].upper() if response else ""

    def score(self, prediction: str, sample: BenchmarkSample) -> bool:
        return prediction.upper().strip() == sample.ground_truth.upper().strip()
