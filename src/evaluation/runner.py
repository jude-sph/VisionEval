"""Main evaluation runner: iterates over benchmark x condition pairs."""

import time
import logging
from typing import Optional
from tqdm import tqdm

from src.benchmarks.base import Benchmark
from src.image_conditions.base import ImageCondition
from src.evaluation.results_store import ResultsStore
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

# How often to log a progress summary (every N questions)
PROGRESS_LOG_INTERVAL = 50


def run_evaluation(
    model,
    tokenizer,
    image_processor,
    benchmark: Benchmark,
    condition: ImageCondition,
    results_store: ResultsStore,
    max_samples: Optional[int] = None,
    conv_mode: str = "llama_3",
    max_new_tokens: int = 128,
) -> dict:
    """Run a single benchmark under a single image condition.

    Supports checkpoint/resume: skips questions already in the results store.

    Returns:
        Aggregated metrics dict.
    """
    from src.model.inference import run_inference

    # Load benchmark data
    benchmark.load(max_samples=max_samples)
    dataset_for_swap = benchmark.get_raw_dataset()

    # Check which questions are already done
    completed = results_store.get_completed_ids(benchmark.name, condition.name)
    logger.info(
        f"Running {benchmark.name} / {condition.name}: "
        f"{len(benchmark)} total, {len(completed)} already done"
    )

    all_results = results_store.load_results(benchmark.name, condition.name)
    skipped = 0
    processed = 0
    correct_count = sum(1 for r in all_results if r.get("correct"))
    total_inference_ms = sum(r.get("inference_time_ms", 0) for r in all_results)
    run_start = time.time()

    for sample in tqdm(
        benchmark,
        total=len(benchmark),
        desc=f"{benchmark.name}/{condition.name}",
    ):
        if sample.question_id in completed:
            skipped += 1
            continue

        # Apply image condition
        transformed_image = condition.apply(
            image=sample.image,
            sample={"question_id": sample.question_id, "_index": sample.metadata.get("_index"), **sample.metadata},
            dataset_images=dataset_for_swap,
        )

        # Format question
        question_text = benchmark.format_question(sample)

        # Run inference
        start_time = time.time()
        raw_response = run_inference(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            question=question_text,
            image=transformed_image,
            conv_mode=conv_mode,
            max_new_tokens=max_new_tokens,
        )
        inference_ms = (time.time() - start_time) * 1000

        # Extract and score answer
        prediction = benchmark.extract_answer(raw_response, sample)
        correct = benchmark.score(prediction, sample)

        result = {
            "question_id": sample.question_id,
            "benchmark": benchmark.name,
            "condition": condition.name,
            "question": sample.question[:200],
            "raw_response": raw_response[:500],
            "prediction": prediction,
            "ground_truth": sample.ground_truth,
            "correct": correct,
            "inference_time_ms": round(inference_ms, 1),
        }

        # Include extra metadata for VQA accuracy
        if sample.metadata and "all_answers" in sample.metadata:
            result["all_answers"] = sample.metadata["all_answers"]

        results_store.append_result(benchmark.name, condition.name, result)
        all_results.append(result)

        processed += 1
        if correct:
            correct_count += 1
        total_inference_ms += inference_ms

        # Periodic progress logging (visible in log files even without tqdm)
        if processed % PROGRESS_LOG_INTERVAL == 0:
            done = len(all_results)
            total = len(benchmark)
            acc = correct_count / done * 100 if done > 0 else 0
            avg_ms = total_inference_ms / done if done > 0 else 0
            remaining = total - done
            eta_s = remaining * avg_ms / 1000
            elapsed = time.time() - run_start
            logger.info(
                f"Progress {benchmark.name}/{condition.name}: "
                f"{done}/{total} ({done/total*100:.1f}%), "
                f"acc={acc:.1f}%, avg={avg_ms:.0f}ms/q, "
                f"ETA={eta_s/60:.0f}min, elapsed={elapsed/60:.1f}min"
            )

    if skipped > 0:
        logger.info(f"Skipped {skipped} already-completed questions")

    # Compute metrics
    metrics = compute_metrics(all_results, benchmark.scoring_method)
    metrics["benchmark"] = benchmark.name
    metrics["condition"] = condition.name
    metrics["num_samples"] = len(all_results)
    metrics["total_inference_ms"] = round(total_inference_ms, 1)

    logger.info(f"Results for {benchmark.name}/{condition.name}: {metrics}")
    return metrics
