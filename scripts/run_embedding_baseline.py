"""Evaluate cached Qwen embeddings with group-aware OOF logistic regression."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.evaluation import (
    apply_thresholds,
    build_folds,
    competition_metrics,
    tune_thresholds,
)
from src.utils_data_prep import read_dataframe
from src.utils_logreg import ProductQualityPredictor
from src.wandb_tracking import WandbTracker, add_wandb_arguments

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLASSIFIER_PATH = PROJECT_DIR / "baseline_qwen3vl_bf16.joblib"
CLASSIFIER_NPZ_PATH = PROJECT_DIR / "baseline_qwen3vl_bf16.npz"


def append_experiment(path: Path, row: dict[str, object]) -> None:
    experiments = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([experiments, pd.DataFrame([row])], ignore_index=True).to_csv(
        path, index=False
    )


def _unique_positive(values: list[float]) -> list[float]:
    result = []
    for value in values:
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"C values must be positive, got {number}")
        if number not in result:
            result.append(number)
    return result


def _backup_artifact(path: Path, backup_dir: Path, timestamp: str) -> Path | None:
    if not path.is_file():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.stem}_{timestamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    return backup_path


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
    parser.add_argument(
        "--classifier-path", type=Path, default=CLASSIFIER_PATH
    )
    parser.add_argument(
        "--classifier-npz-path", type=Path, default=CLASSIFIER_NPZ_PATH
    )
    parser.add_argument("--experiment-id", default="E10_qwen_embedding_mps")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        help="Search several C values with the same group-aware folds",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2-normalize embeddings before LogisticRegression",
    )
    parser.add_argument(
        "--search-normalization",
        action="store_true",
        help="Evaluate both raw and L2-normalized embeddings",
    )
    add_wandb_arguments(parser)
    args = parser.parse_args()

    if args.n_splits < 2:
        parser.error("--n-splits must be at least 2")
    try:
        c_grid = _unique_positive(args.c_grid or [args.c])
    except ValueError as error:
        parser.error(str(error))
    normalize_grid = [False, True] if args.search_normalization else [args.normalize]
    configurations = [
        {"c": c_value, "normalize": normalize}
        for normalize in normalize_grid
        for c_value in c_grid
    ]

    started = datetime.now(timezone.utc)
    df = read_dataframe(args.data)
    if not args.embeddings.is_file():
        raise FileNotFoundError(
            f"embeddings not found: {args.embeddings}; run extract_embeddings_mps first"
        )
    ids_path = args.embeddings.with_suffix(".ids.csv")
    if not ids_path.is_file():
        raise FileNotFoundError(f"embedding id manifest not found: {ids_path}")
    embeddings = np.load(args.embeddings)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("cached embeddings must be a finite two-dimensional matrix")
    embedding_ids = pd.read_csv(ids_path)["id"]
    if len(embeddings) != len(df) or not np.array_equal(embedding_ids, df["id"]):
        raise ValueError("cached embeddings do not match the training CSV ids and order")

    folds = build_folds(df, n_splits=args.n_splits, random_state=args.random_state)
    df = df.merge(folds[["id", "fold"]], on="id", validate="one_to_one")
    labels = df["label"].to_numpy()
    categories = df["category"].astype(str).to_numpy()
    fold_values = df["fold"].to_numpy()

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
            "c_grid": c_grid,
            "normalize_grid": normalize_grid,
            "embedding_path": str(args.embeddings.resolve()),
            "embedding_rows": embeddings.shape[0],
            "embedding_dimensions": embeddings.shape[1],
        },
    )

    candidate_results: list[dict[str, object]] = []
    for config_index, config in enumerate(configurations, start=1):
        c_value = float(config["c"])
        normalize = bool(config["normalize"])
        print(
            f"Configuration {config_index}/{len(configurations)}: "
            f"C={c_value:g}, normalize={normalize}"
        )
        oof_probability = np.full(len(df), np.nan)
        fold_runtimes: list[float] = []
        for fold in range(args.n_splits):
            train_mask = fold_values != fold
            valid_mask = ~train_mask
            print(
                f"  Fold {fold}: train={int(train_mask.sum()):,}, "
                f"valid={int(valid_mask.sum()):,}"
            )
            fold_started = perf_counter()
            predictor = ProductQualityPredictor().fit(
                embeddings[train_mask],
                labels[train_mask],
                categories[train_mask],
                c=c_value,
                normalize=normalize,
                random_state=args.random_state,
            )
            valid_probability, _ = predictor.predict(
                embeddings[valid_mask], categories[valid_mask]
            )
            oof_probability[valid_mask] = valid_probability
            fold_runtimes.append(perf_counter() - fold_started)

        if not np.isfinite(oof_probability).all():
            raise RuntimeError("OOF prediction contains missing or invalid probabilities")
        metrics_at_05 = competition_metrics(
            labels, oof_probability >= 0.5, categories
        )
        thresholds = tune_thresholds(labels, oof_probability, categories)
        predictions = apply_thresholds(oof_probability, categories, thresholds)
        metrics_tuned = competition_metrics(labels, predictions, categories)
        candidate_results.append(
            {
                "c": c_value,
                "normalize": normalize,
                "oof_probability": oof_probability,
                "fold_runtimes": fold_runtimes,
                "thresholds": thresholds,
                "metrics_at_0_5": metrics_at_05,
                "metrics_tuned": metrics_tuned,
            }
        )
        print(f"  tuned mean F1={metrics_tuned['mean_f1']:.6f}")

    selected_configs: dict[str, dict[str, object]] = {}
    selected_probability = np.full(len(df), np.nan)
    thresholds: dict[str, float] = {}
    for category in sorted(np.unique(categories)):
        best_candidate = max(
            candidate_results,
            key=lambda item: item["metrics_tuned"]["per_category"][category]["f1"],
        )
        mask = categories == category
        selected_probability[mask] = best_candidate["oof_probability"][mask]
        thresholds[category] = float(best_candidate["thresholds"][category])
        selected_configs[category] = {
            "c": float(best_candidate["c"]),
            "normalize": bool(best_candidate["normalize"]),
            "threshold": thresholds[category],
            "oof_f1": float(
                best_candidate["metrics_tuned"]["per_category"][category]["f1"]
            ),
        }

    predictions = apply_thresholds(selected_probability, categories, thresholds)
    metrics_at_05 = competition_metrics(
        labels, selected_probability >= 0.5, categories
    )
    metrics = competition_metrics(labels, predictions, categories)

    for fold in range(args.n_splits):
        valid_mask = fold_values == fold
        fold_metrics = competition_metrics(
            labels[valid_mask],
            selected_probability[valid_mask] >= 0.5,
            categories[valid_mask],
        )
        tracker.log_fold(
            fold=fold,
            train_rows=int((~valid_mask).sum()),
            valid_rows=int(valid_mask.sum()),
            runtime_seconds=float(
                sum(item["fold_runtimes"][fold] for item in candidate_results)
            ),
            metrics=fold_metrics,
        )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "oof").mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    folds_path = args.reports_dir / "folds.csv"
    folds.to_csv(folds_path, index=False)
    oof_path = args.reports_dir / "oof" / f"{args.experiment_id}.csv"
    pd.DataFrame(
        {
            "id": df["id"],
            "category": categories,
            "label": labels,
            "fold": fold_values,
            "probability": selected_probability,
            "prediction": predictions,
            "selected_c": [selected_configs[c]["c"] for c in categories],
            "selected_normalize": [
                selected_configs[c]["normalize"] for c in categories
            ],
            "is_error": labels != predictions,
        }
    ).to_csv(oof_path, index=False)

    final_predictor = ProductQualityPredictor().fit(
        embeddings,
        labels,
        categories,
        c={category: values["c"] for category, values in selected_configs.items()},
        normalize={
            category: values["normalize"]
            for category, values in selected_configs.items()
        },
        random_state=args.random_state,
    )
    final_predictor.set_thresholds(thresholds)
    args.classifier_path.parent.mkdir(parents=True, exist_ok=True)
    args.classifier_npz_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = args.artifacts_dir / "backups"
    backup_paths = [
        path
        for path in (
            _backup_artifact(args.classifier_path, backup_dir, timestamp),
            _backup_artifact(args.classifier_npz_path, backup_dir, timestamp),
        )
        if path is not None
    ]
    final_predictor.save(args.classifier_path)
    final_predictor.export_npz(args.classifier_npz_path)

    candidate_reports = [
        {
            "c": item["c"],
            "normalize": item["normalize"],
            "runtime_seconds": float(sum(item["fold_runtimes"])),
            "thresholds": item["thresholds"],
            "metrics_at_0_5": item["metrics_at_0_5"],
            "metrics_tuned": item["metrics_tuned"],
        }
        for item in candidate_results
    ]
    runtime = (datetime.now(timezone.utc) - started).total_seconds()
    summary = {
        "experiment_id": args.experiment_id,
        "rows": len(df),
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "runtime_seconds": runtime,
        "embedding_path": str(args.embeddings.resolve()),
        "embedding_shape": list(embeddings.shape),
        "search": {
            "c_grid": c_grid,
            "normalize_grid": normalize_grid,
            "candidates": candidate_reports,
        },
        "selected_configs": selected_configs,
        "thresholds": thresholds,
        "metrics_at_0_5": metrics_at_05,
        "metrics_tuned": metrics,
        "oof_path": str(oof_path.resolve()),
        "folds_path": str(folds_path.resolve()),
        "artifact_path": str(args.classifier_path.resolve()),
        "npz_artifact_path": str(args.classifier_npz_path.resolve()),
        "backup_paths": [str(path.resolve()) for path in backup_paths],
        "predictor_summary": final_predictor.summary(),
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
        "selected_configs": json.dumps(
            selected_configs, ensure_ascii=False, sort_keys=True
        ),
        "candidate_count": len(candidate_reports),
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
    tracker.log_model_selection(
        selected_configs=selected_configs,
        candidates=candidate_reports,
    )
    if args.wandb_log_artifacts:
        tracker.log_artifacts(
            experiment_id=args.experiment_id,
            report_files=[summary_path, oof_path, folds_path],
            model_paths=[args.classifier_path, args.classifier_npz_path],
        )
    tracker.finish()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Selected: {json.dumps(selected_configs, ensure_ascii=False)}")
    print(f"Classifier: {final_predictor.summary()}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
