"""Cross-validation and competition metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from src.utils_data_prep import build_group_key


def build_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create stratified folds without splitting duplicate cards."""
    groups = build_group_key(df)
    strata = df["category"].astype(str) + "__" + df["label"].astype(str)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = np.empty(len(df), dtype=np.int8)
    for fold, (_, valid_index) in enumerate(splitter.split(df, strata, groups)):
        folds[valid_index] = fold

    return pd.DataFrame({"id": df["id"], "fold": folds})


def competition_metrics(y_true, y_pred, categories) -> dict[str, object]:
    """Calculate precision, recall and F1 for each category and mean category F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    categories = np.asarray(categories).astype(str)

    per_category = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[mask],
            y_pred[mask],
            average="binary",
            zero_division=0,
        )
        per_category[category] = {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "count": int(mask.sum()),
        }

    mean_f1 = np.mean([metrics["f1"] for metrics in per_category.values()])
    return {"per_category": per_category, "mean_f1": float(mean_f1)}


def tune_thresholds(
    y_true,
    probabilities,
    categories,
    grid=None,
) -> dict[str, float]:
    """Find the best F1 threshold separately for each category."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    categories = np.asarray(categories).astype(str)
    grid = np.asarray(grid if grid is not None else np.linspace(0.01, 0.99, 99))

    result = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        scores = [
            f1_score(
                y_true[mask],
                probabilities[mask] >= threshold,
                zero_division=0,
            )
            for threshold in grid
        ]
        best_score = max(scores)
        plateau = grid[np.isclose(scores, best_score, rtol=0.0, atol=1e-12)]
        result[category] = float(np.median(plateau))

    return result


def apply_thresholds(probabilities, categories, thresholds) -> np.ndarray:
    """Convert probabilities to predictions using a threshold per category."""
    probabilities = np.asarray(probabilities)
    category_thresholds = np.array(
        [thresholds[str(category)] for category in categories]
    )
    return (probabilities >= category_thresholds).astype(np.int8)
