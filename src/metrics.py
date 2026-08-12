"""Competition metrics and threshold selection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _as_1d(name: str, values: Sequence) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def competition_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    categories: Sequence[str],
) -> dict[str, object]:
    # считаем f1 по категориям, затем усредняем
    true = _as_1d("y_true", y_true)
    pred = _as_1d("y_pred", y_pred)
    category_values = _as_1d("categories", categories)

    if not (len(true) == len(pred) == len(category_values)):
        raise ValueError(
            "y_true, y_pred and categories must have equal lengths: "
            f"{len(true)}, {len(pred)}, {len(category_values)}"
        )
    if len(true) == 0:
        raise ValueError("metric input must not be empty")

    invalid_true = set(np.unique(true)) - {0, 1}
    invalid_pred = set(np.unique(pred)) - {0, 1}
    if invalid_true or invalid_pred:
        raise ValueError(
            f"labels and predictions must be binary; got labels={invalid_true}, "
            f"predictions={invalid_pred}"
        )

    per_category: dict[str, dict[str, float | int]] = {}
    for category in sorted(map(str, np.unique(category_values))):
        mask = category_values.astype(str) == category
        category_true = true[mask]
        category_pred = pred[mask]
        per_category[category] = {
            "f1": float(f1_score(category_true, category_pred, zero_division=0)),
            "precision": float(
                precision_score(category_true, category_pred, zero_division=0)
            ),
            "recall": float(recall_score(category_true, category_pred, zero_division=0)),
            "count": int(mask.sum()),
        }

    mean_f1 = float(np.mean([item["f1"] for item in per_category.values()]))
    return {"per_category": per_category, "mean_f1": mean_f1}


def tune_thresholds(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    categories: Sequence[str],
    grid: Sequence[float] | None = None,
) -> dict[str, float]:
    """Select an F1-optimal threshold independently for every category.

    If several thresholds have equal F1, the median of the best plateau is used.
    """
    true = _as_1d("y_true", y_true)
    probs = _as_1d("probabilities", probabilities).astype(float)
    category_values = _as_1d("categories", categories).astype(str)
    if not (len(true) == len(probs) == len(category_values)):
        raise ValueError("y_true, probabilities and categories must have equal lengths")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain non-finite values")

    thresholds = np.asarray(
        list(grid) if grid is not None else np.linspace(0.01, 0.99, 99),
        dtype=float,
    )
    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("threshold grid must be a non-empty one-dimensional sequence")

    result: dict[str, float] = {}
    for category in sorted(np.unique(category_values)):
        mask = category_values == category
        scores = np.asarray(
            [
                f1_score(true[mask], probs[mask] >= threshold, zero_division=0)
                for threshold in thresholds
            ]
        )
        best = thresholds[np.isclose(scores, scores.max())]
        result[str(category)] = float(np.median(best))
    return result


def apply_thresholds(
    probabilities: Sequence[float],
    categories: Sequence[str],
    thresholds: dict[str, float],
) -> np.ndarray:
    """Convert probabilities to binary predictions using category thresholds."""
    probs = _as_1d("probabilities", probabilities).astype(float)
    category_values = _as_1d("categories", categories).astype(str)
    if len(probs) != len(category_values):
        raise ValueError("probabilities and categories must have equal lengths")

    missing = set(np.unique(category_values)) - set(thresholds)
    if missing:
        raise ValueError(f"missing thresholds for categories: {sorted(missing)}")
    return np.asarray(
        [prob >= thresholds[category] for prob, category in zip(probs, category_values)],
        dtype=np.int8,
    )

