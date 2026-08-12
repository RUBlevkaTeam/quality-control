"""Train and evaluate the local TF-IDF baseline with OOF predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.folds import build_folds
from src.metrics import apply_thresholds, competition_metrics, tune_thresholds
from src.text_baseline import TextBaselineConfig, TextQualityPredictor
from src.wandb_tracking import WandbTracker, add_wandb_arguments

PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "content/content_ecup/data.csv",
        help="Training CSV containing category and label",
    )
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_DIR / "reports")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_DIR / "artifacts")
    parser.add_argument("--experiment-id", help="Stable experiment name")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--word-max-features", type=int, default=40_000)
    parser.add_argument("--char-max-features", type=int, default=60_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--class-weight", choices=("balanced", "none"), default="balanced"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 3 folds and fewer features for a fast smoke test",
    )
    add_wandb_arguments(parser)
    return parser.parse_args()


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"training CSV not found: {path}")
    df = pd.read_csv(path)
    junk = [column for column in df.columns if str(column).startswith("Unnamed:")]
    if junk:
        df = df.drop(columns=junk)
    required = {"id", "name", "description", "category", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"training CSV is missing columns: {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError("training CSV contains duplicate id values")
    invalid_labels = set(df["label"].dropna().unique()) - {0, 1}
    if invalid_labels or df["label"].isna().any():
        raise ValueError(f"label must contain only 0/1, got: {invalid_labels}")
    return df.reset_index(drop=True)


def append_experiment(path: Path, row: dict[str, object]) -> None:
    frame = pd.DataFrame([row])
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    experiment_id = args.experiment_id or started_at.strftime("E%Y%m%d_%H%M%S")

    n_splits = 3 if args.quick else args.n_splits
    word_max_features = 10_000 if args.quick else args.word_max_features
    char_max_features = 20_000 if args.quick else args.char_max_features
    config = TextBaselineConfig(
        word_max_features=word_max_features,
        char_max_features=char_max_features,
        min_df=args.min_df,
        c=args.c,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        random_state=args.random_state,
    )

    df = load_training_data(args.data)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "oof").mkdir(parents=True, exist_ok=True)

    folds = build_folds(df, n_splits=n_splits, random_state=args.random_state)
    folds.to_csv(args.reports_dir / "folds.csv", index=False)
    df = df.merge(folds[["id", "fold"]], on="id", how="left", validate="one_to_one")

    tracker = WandbTracker.start(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=experiment_id,
        group="tfidf-baseline",
        tags=["tfidf", "oof", *args.wandb_tags],
        config={
            "experiment_id": experiment_id,
            "model": "word-char-tfidf-logreg",
            "rows": len(df),
            "n_splits": n_splits,
            "data_path": str(args.data.resolve()),
            **asdict(config),
        },
    )

    oof_probability = np.full(len(df), np.nan, dtype=np.float64)
    print(f"Rows: {len(df):,}; folds: {n_splits}; experiment: {experiment_id}")
    for fold in range(n_splits):
        train = df[df["fold"] != fold]
        valid = df[df["fold"] == fold]
        print(f"Fold {fold}: train={len(train):,}, valid={len(valid):,}")
        fold_started = perf_counter()
        predictor = TextQualityPredictor(config).fit(train)
        valid_probability = predictor.predict_proba(valid)
        oof_probability[valid.index] = valid_probability
        fold_metrics = competition_metrics(
            valid["label"].to_numpy(),
            valid_probability >= 0.5,
            valid["category"].astype(str).to_numpy(),
        )
        tracker.log_fold(
            fold=fold,
            train_rows=len(train),
            valid_rows=len(valid),
            runtime_seconds=perf_counter() - fold_started,
            metrics=fold_metrics,
        )

    if not np.isfinite(oof_probability).all():
        raise RuntimeError("OOF probabilities are incomplete")

    categories = df["category"].astype(str).to_numpy()
    labels = df["label"].to_numpy()
    metrics_at_05 = competition_metrics(labels, oof_probability >= 0.5, categories)
    thresholds = tune_thresholds(labels, oof_probability, categories)
    predictions = apply_thresholds(oof_probability, categories, thresholds)
    tuned_metrics = competition_metrics(labels, predictions, categories)

    oof = df[["id", "category", "label", "fold", "name"]].copy()
    oof["probability"] = oof_probability
    oof["prediction"] = predictions
    oof["is_error"] = oof["label"] != oof["prediction"]
    oof_path = args.reports_dir / "oof" / f"{experiment_id}.csv"
    oof.to_csv(oof_path, index=False)
    oof[oof["is_error"]].sort_values("probability").to_csv(
        args.reports_dir / "error_analysis.csv", index=False
    )

    print("Fitting final model on all rows...")
    final_predictor = TextQualityPredictor(config).fit(df)
    final_predictor.thresholds = thresholds
    artifact_path = args.artifacts_dir / "text_baseline.joblib"
    joblib.dump(final_predictor, artifact_path)

    finished_at = datetime.now(timezone.utc)
    summary = {
        "experiment_id": experiment_id,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "runtime_seconds": (finished_at - started_at).total_seconds(),
        "data_path": str(args.data.resolve()),
        "rows": len(df),
        "n_splits": n_splits,
        "config": asdict(config),
        "thresholds": thresholds,
        "metrics_at_0_5": metrics_at_05,
        "metrics_tuned": tuned_metrics,
        "oof_path": str(oof_path.resolve()),
        "artifact_path": str(artifact_path.resolve()),
    }
    summary_path = args.reports_dir / f"{experiment_id}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    experiment_row: dict[str, object] = {
        "experiment_id": experiment_id,
        "finished_at_utc": finished_at.isoformat(),
        "n_splits": n_splits,
        "word_max_features": word_max_features,
        "char_max_features": char_max_features,
        "c": config.c,
        "class_weight": config.class_weight,
        "thresholds": json.dumps(thresholds, ensure_ascii=False, sort_keys=True),
        "mean_f1_at_0_5": metrics_at_05["mean_f1"],
        "mean_f1_tuned": tuned_metrics["mean_f1"],
        "runtime_seconds": summary["runtime_seconds"],
    }
    for category, values in tuned_metrics["per_category"].items():
        experiment_row[f"f1_{category}"] = values["f1"]
        experiment_row[f"precision_{category}"] = values["precision"]
        experiment_row[f"recall_{category}"] = values["recall"]
    append_experiment(args.reports_dir / "experiments.csv", experiment_row)

    tracker.log_final(
        metrics_at_05=metrics_at_05,
        metrics_tuned=tuned_metrics,
        thresholds=thresholds,
        runtime_seconds=float(summary["runtime_seconds"]),
        error_count=int(oof["is_error"].sum()),
    )
    if args.wandb_log_artifacts:
        tracker.log_artifacts(
            experiment_id=experiment_id,
            report_files=[
                summary_path,
                oof_path,
                args.reports_dir / "error_analysis.csv",
                args.reports_dir / "folds.csv",
            ],
            model_path=artifact_path,
        )
    tracker.finish()

    print(json.dumps(tuned_metrics, ensure_ascii=False, indent=2))
    print(f"Thresholds: {thresholds}")
    print(f"Summary: {summary_path}")
    print(f"Model: {artifact_path}")


if __name__ == "__main__":
    main()
