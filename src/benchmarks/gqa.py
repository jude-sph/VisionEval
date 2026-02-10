"""GQA benchmark adapter — compositional visual reasoning from scene graphs."""

import logging
from typing import Iterator
from datasets import load_dataset
from PIL import Image

from .base import Benchmark, BenchmarkSample

logger = logging.getLogger(__name__)


class GQABenchmark(Benchmark):
    name = "gqa"
    scoring_method = "exact_match"

    def load(self, max_samples: int | None = None) -> None:
        # lmms-lab/GQA splits images and instructions into separate configs
        images_ds = load_dataset(
            "lmms-lab/GQA", "testdev_balanced_images", split="testdev"
        )
        instructions_ds = load_dataset(
            "lmms-lab/GQA", "testdev_balanced_instructions", split="testdev"
        )

        # Build image lookup: imageId -> PIL Image
        image_lookup = {}
        for row in images_ds:
            image_lookup[row["id"]] = row["image"]

        if max_samples:
            instructions_ds = instructions_ds.select(
                range(min(max_samples, len(instructions_ds)))
            )

        self._instructions = instructions_ds
        self._image_lookup = image_lookup
        logger.info(
            f"GQA loaded: {len(instructions_ds)} questions, "
            f"{len(image_lookup)} unique images"
        )

    def __len__(self) -> int:
        return len(self._instructions)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        for idx, row in enumerate(self._instructions):
            image_id = row.get("imageId", "")
            image = self._image_lookup.get(image_id)
            if image is None:
                continue
            if not isinstance(image, Image.Image):
                continue

            yield BenchmarkSample(
                question_id=f"gqa_{row.get('id', idx)}",
                question=row.get("question", ""),
                image=image.convert("RGB"),
                ground_truth=row.get("answer", "").lower().strip(),
                metadata={"_index": idx},
            )

    def get_raw_dataset(self):
        """Return image lookup for wrong_image condition."""
        # Convert to list of dicts with 'image' key for wrong_image compatibility
        return [{"image": img} for img in self._image_lookup.values()]

    def format_question(self, sample: BenchmarkSample) -> str:
        return f"{sample.question}\nAnswer the question using a single word or phrase."

    def extract_answer(self, response: str, sample: BenchmarkSample) -> str:
        response = response.strip().lower()
        response = response.split("\n")[0].strip()
        response = response.rstrip(".")
        return response

    def score(self, prediction: str, sample: BenchmarkSample) -> bool:
        return prediction.lower().strip() == sample.ground_truth.lower().strip()
