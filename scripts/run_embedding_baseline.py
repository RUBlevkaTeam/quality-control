"""Evaluate cached Qwen embeddings with group-aware OOF logistic regression."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from scripts.run_text_baseline import append_experiment, load_training_data
from src.folds import build_folds
from src.metrics import apply_thresholds, competition_metrics, tune_thresholds
from src.wandb_tracking import WandbTracker, add_wandb_arguments

PROJECT_DIR = Path(__file__).resolve().parents[1]


def fit_category_models(
    embeddings: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    c: float,
    random_state: int,
) -> dict[str, LogisticRegression]:
    models: dict[str, LogisticRegression] = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=1_000,
            solver="liblinear",
            random_state=random_state,
        )
        model.fit(embeddings[mask], labels[mask])
        models[str(category)] = model
    return models


def predict_category_models(
    models: dict[str, LogisticRegression],
    embeddings: np.ndarray,
    categories: np.ndarray,
) -> np.ndarray:
    probabilities = np.zeros(len(embeddings), dtype=np.float64)
    for category in np.unique(categories):
        mask = categories == category
        probabilities[mask] = models[str(category)].predict_proba(embeddings[mask])[:, 1]
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=PROJECT_DIR / "content/content_ecup/data.csv"
    )
    parser.add_argument(
        "--embeddings", type=Path, default=PROJECT_DIR / "cache/qwen3_vl_mps.npy"
    )
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_DIR / "reports")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_DIR / "artifacts")
    parser.add_argument("--experiment-id", default="E10_qwen_embedding_mps")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    add_wandb_arguments(parser)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    df = load_training_data(args.data)
    if not args.embeddings.is_file():
        raise FileNotFoundError(
            f"embeddings not found: {args.embeddings}; run extract_embeddings_mps first"
        )
    ids_path = args.embeddings.with_suffix(".ids.csv")
    if not ids_path.is_file():
        raise FileNotFoundError(f"embedding id manifest not found: {ids_path}")
    embeddings = np.load(args.embeddings)
    embedding_ids = pd.read_csv(ids_path)["id"]
    if len(embeddings) != len(df) or not np.array_equal(embedding_ids, df["id"]):
        raise ValueError("cached embeddings do not match the training CSV ids and order")
    embeddings = normalize(embeddings, norm="l2", copy=False)

    folds = build_folds(df, n_splits=args.n_splits, random_state=args.random_state)
    df = df.merge(folds[["id", "fold"]], on="id", validate="one_to_one")
    labels = df["label"].to_numpy()
    categories = df["category"].astype(str).to_numpy()
    oof_probability = np.full(len(df), np.nan)

    tracker = WandbTracker.start(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.experiment_id,
        group="qwen-embedding-baseline",
        tags=["qwen", "embedding", "mps", "oof", *args.wandb_tags],
        config={
            "experiment_id": args.experiment_id,
            "model": "qwen-embedding-logreg",
            "rows": len(df),
            "n_splits": args.n_splits,
            "random_state": args.random_state,
            "c": args.c,
            "embedding_path": str(args.embeddings.resolve()),
            "embedding_rows": embeddings.shape[0],
            "embedding_dimensions": embeddings.shape[1],
        },
    )

    for fold in range(args.n_splits):
        train_mask = df["fold"].to_numpy() != fold
        valid_mask = ~train_mask
        print(
            f"Fold {fold}: train={int(train_mask.sum()):,}, valid={int(valid_mask.sum()):,}"
        )
        fold_started = perf_counter()
        models = fit_category_models(
            embeddings[train_mask],
            labels[train_mask],
            categories[train_mask],
            args.c,
            args.random_state,
        )
        valid_probability = predict_category_models(
            models, embeddings[valid_mask], categories[valid_mask]
        )
        oof_probability[valid_mask] = valid_probability
        fold_metrics = competition_metrics(
            labels[valid_mask], valid_probability >= 0.5, categories[valid_mask]
        )
        tracker.log_fold(
            fold=fold,
            train_rows=int(train_mask.sum()),
            valid_rows=int(valid_mask.sum()),
            runtime_seconds=perf_counter() - fold_started,
            metrics=fold_metrics,
        )

    metrics_at_05 = competition_metrics(
        labels, oof_probability >= 0.5, categories
    )
    thresholds = tune_thresholds(labels, oof_probability, categories)
    predictions = apply_thresholds(oof_probability, categories, thresholds)
    metrics = competition_metrics(labels, predictions, categories)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "oof").mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.reports_dir / "oof" / f"{args.experiment_id}.csv"
    pd.DataFrame(
        {
            "id": df["id"],
            "category": categories,
            "label": labels,
            "fold": df["fold"],
            "probability": oof_probability,
            "prediction": predictions,
        }
    ).to_csv(oof_path, index=False)

    final_models = fit_category_models(
        embeddings, labels, categories, args.c, args.random_state
    )
    artifact_path = args.artifacts_dir / "embedding_baseline.joblib"
    joblib.dump(
        {"models": final_models, "thresholds": thresholds, "normalized": True},
        artifact_path,
    )
    runtime = (datetime.now(timezone.utc) - started).total_seconds()
    summary = {
        "experiment_id": args.experiment_id,
        "runtime_seconds": runtime,
        "embedding_path": str(args.embeddings.resolve()),
        "embedding_shape": list(embeddings.shape),
        "c": args.c,
        "thresholds": thresholds,
        "metrics_at_0_5": metrics_at_05,
        "metrics_tuned": metrics,
        "oof_path": str(oof_path.resolve()),
        "artifact_path": str(artifact_path.resolve()),
    }
    summary_path = args.reports_dir / f"{args.experiment_id}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    row: dict[str, object] = {
        "experiment_id": args.experiment_id,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_splits": args.n_splits,
        "model": "qwen_embedding_mps_logreg",
        "c": args.c,
        "thresholds": json.dumps(thresholds, ensure_ascii=False, sort_keys=True),
        "mean_f1_at_0_5": metrics_at_05["mean_f1"],
        "mean_f1_tuned": metrics["mean_f1"],
        "runtime_seconds": runtime,
    }
    for category, values in metrics["per_category"].items():
        row[f"f1_{category}"] = values["f1"]
        row[f"precision_{category}"] = values["precision"]
        row[f"recall_{category}"] = values["recall"]
    append_experiment(args.reports_dir / "experiments.csv", row)
    tracker.log_final(
        metrics_at_05=metrics_at_05,
        metrics_tuned=metrics,
        thresholds=thresholds,
        runtime_seconds=runtime,
        error_count=int(np.sum(labels != predictions)),
    )
    if args.wandb_log_artifacts:
        tracker.log_artifacts(
            experiment_id=args.experiment_id,
            report_files=[summary_path, oof_path],
            model_path=artifact_path,
        )
    tracker.finish()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Summary: {args.reports_dir / f'{args.experiment_id}.json'}")


if __name__ == "__main__":
    main()
