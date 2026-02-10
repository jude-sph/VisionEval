"""Scoring metrics for benchmark evaluation."""

from sklearn.metrics import f1_score, precision_score, recall_score


def _filter_errors(results: list[dict]) -> list[dict]:
    """Exclude error samples from metric calculations."""
    return [r for r in results if not r.get("error")]


def compute_accuracy(results: list[dict]) -> float:
    """Simple accuracy: fraction of correct predictions (excluding errors)."""
    results = _filter_errors(results)
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["correct"])
    return correct / len(results)


def compute_binary_metrics(results: list[dict]) -> dict[str, float]:
    """Accuracy, F1, precision, recall for binary (yes/no) tasks like POPE."""
    results = _filter_errors(results)
    if not results:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "yes_ratio": 0.0}

    y_true = [1 if r["ground_truth"] == "yes" else 0 for r in results]
    y_pred = [1 if r["prediction"] == "yes" else 0 for r in results]

    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    yes_ratio = sum(y_pred) / len(y_pred)

    return {
        "accuracy": accuracy,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "yes_ratio": yes_ratio,
    }


def compute_vqa_accuracy(results: list[dict]) -> float:
    """VQA accuracy with soft matching against multiple annotator answers."""
    results = _filter_errors(results)
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        all_answers = r.get("all_answers", [r["ground_truth"]])
        pred = r["prediction"].lower().strip().rstrip(".")
        count = sum(1 for a in all_answers if a.lower().strip().rstrip(".") == pred)
        total += min(count / 3.0, 1.0)
    return total / len(results)


def compute_metrics(results: list[dict], scoring_method: str) -> dict[str, float]:
    """Compute metrics based on the benchmark's scoring method."""
    if scoring_method == "mc_accuracy":
        return {"accuracy": compute_accuracy(results)}
    elif scoring_method == "binary_accuracy_f1":
        return compute_binary_metrics(results)
    elif scoring_method == "vqa_accuracy":
        return {"accuracy": compute_vqa_accuracy(results)}
    elif scoring_method == "exact_match":
        return {"accuracy": compute_accuracy(results)}
    else:
        return {"accuracy": compute_accuracy(results)}
