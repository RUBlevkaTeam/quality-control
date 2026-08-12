#!/usr/bin/env python3
"""Train and evaluate the compact artifacts used by the ultra submission."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ultra_features import BAD_CATEGORY, FLAMMABLE_CATEGORY, evidence_flags
from src.ultra_model import (
    ULTRA_ARTIFACT_FORMAT_VERSION,
    ULTRA_FEATURE_SCHEMA,
    apply_empirical_absence_veto,
    best_f1_threshold,
    combine_text_and_retrieval,
    fit_text_model,
    predict_text_probabilities,
)
from src.ultra_retrieval import (
    build_retrieval_index,
    hash_dataset_images,
    query_retrieval,
)


DEFAULT_CONFIGS = {
    BAD_CATEGORY: {
        "C": 3.0,
        "class_weight": "balanced",
        "word_max_features": 120_000,
        "char_max_features": 180_000,
        "rule_weight": 2.0,
        "random_state": 2026,
    },
    FLAMMABLE_CATEGORY: {
        "C": 3.0,
        "class_weight": "balanced",
        "word_max_features": 100_000,
        "char_max_features": 160_000,
        "rule_weight": 3.0,
        "random_state": 2026,
    },
}


def _jsonable_config(config: dict) -> dict:
    return {
        key: ({str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else value)
        for key, value in config.items()
    }


def _load_or_hash_images(df: pd.DataFrame, images_root: Path, cache_path: Path):
    if cache_path.exists():
        cached = joblib.load(cache_path)
        if cached.get("ids") == df["id"].astype(str).tolist():
            print(f"Loaded image SHA-256 cache: {cache_path}", flush=True)
            return cached["hashes"]

    print(f"Hashing train images under {images_root} ...", flush=True)
    started = time.perf_counter()
    hashes = hash_dataset_images(df["id"].tolist(), images_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"ids": df["id"].astype(str).tolist(), "hashes": hashes}, cache_path, compress=3)
    print(f"Hashed {sum(map(len, hashes))} unique-per-product images in {time.perf_counter()-started:.1f}s", flush=True)
    return hashes


def _retrieval_oof(df, hashes, train_positions, val_positions):
    train = df.iloc[train_positions]
    train_hashes = [hashes[position] for position in train_positions]
    index = build_retrieval_index(
        train["name"].tolist(),
        train["description"].tolist(),
        train["category"].tolist(),
        train["label"].astype(int).tolist(),
        train_hashes,
        row_indices=list(range(len(train))),
    )

    evidence = []
    for position in val_positions:
        row = df.iloc[position]
        evidence.append(
            query_retrieval(
                index,
                category=row["category"],
                name=row["name"],
                description=row["description"],
                image_hashes=hashes[position],
            )
        )
    return evidence


def _metric_row(y_true, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def train_category_oof(df, hashes, category: str, config: dict, n_splits: int):
    positions = np.flatnonzero(df["category"].astype(str).eq(category).to_numpy())
    subset = df.iloc[positions].reset_index(drop=True)
    y = subset["label"].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2026)

    text_oof = np.zeros(len(subset), dtype=np.float64)
    retrieval_oof: list[dict | None] = [None] * len(subset)
    started = time.perf_counter()

    for fold, (local_train, local_val) in enumerate(splitter.split(subset, y), start=1):
        train_rows = subset.iloc[local_train]
        val_rows = subset.iloc[local_val]
        model = fit_text_model(
            train_rows["name"].tolist(),
            train_rows["description"].tolist(),
            train_rows["category"].tolist(),
            train_rows["label"].astype(int).tolist(),
            config=config,
        )
        text_oof[local_val] = predict_text_probabilities(
            model,
            val_rows["name"].tolist(),
            val_rows["description"].tolist(),
            val_rows["category"].tolist(),
        )

        global_train = positions[local_train]
        global_val = positions[local_val]
        fold_evidence = _retrieval_oof(df, hashes, global_train, global_val)
        for local_position, item in zip(local_val, fold_evidence):
            retrieval_oof[local_position] = item
        print(f"  {category}: fold {fold}/{n_splits} complete", flush=True)

    evidence = [item for item in retrieval_oof if item is not None]
    candidates: list[tuple[str, float, np.ndarray]] = [("text", 0.0, text_oof)]
    candidates.append(
        ("hard", 0.0, combine_text_and_retrieval(text_oof, evidence, mode="hard"))
    )
    for mode in ("tier_a", "tier_a_negative", "tier_a_positive"):
        candidates.append(
            (mode, 0.0, combine_text_and_retrieval(text_oof, evidence, mode=mode))
        )
    for weight in (0.5, 1.0, 1.5, 2.0, 3.0):
        candidates.append(
            (
                "hybrid",
                weight,
                combine_text_and_retrieval(
                    text_oof, evidence, mode="hybrid", weight=weight
                ),
            )
        )

    results = []
    best = None
    flag_rows = [
        evidence_flags(row["name"], row["description"])
        for _, row in subset.iterrows()
    ]
    for mode, weight, probabilities in candidates:
        probabilities, _ = apply_empirical_absence_veto(
            probabilities,
            subset["category"].tolist(),
            flag_rows,
        )
        threshold, f1 = best_f1_threshold(y, probabilities)
        metrics = _metric_row(y, probabilities >= threshold)
        row = {
            "mode": mode,
            "weight": weight,
            "threshold": threshold,
            **metrics,
        }
        results.append(row)
        if best is None or f1 > best[0]:
            best = (f1, mode, weight, threshold, probabilities)

    assert best is not None
    hard_coverage = np.mean([item["hard_label"] is not None for item in evidence])
    print(
        f"  {category}: best={best[1]} weight={best[2]} threshold={best[3]:.6f} "
        f"F1={best[0]:.6f}, hard retrieval coverage={hard_coverage:.3%}, "
        f"elapsed={time.perf_counter()-started:.1f}s",
        flush=True,
    )
    return {
        "category": category,
        "y": y,
        "text_oof": text_oof,
        "retrieval_oof": evidence,
        "selected_probabilities": best[4],
        "selected_policy": {"mode": best[1], "weight": best[2], "threshold": best[3]},
        "candidates": results,
        "hard_retrieval_coverage": float(hard_coverage),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=ROOT / "content/content_ecup/data.csv")
    parser.add_argument("--images-root", type=Path, default=ROOT / "content/content_ecup/images")
    parser.add_argument("--output", type=Path, default=ROOT / "ultra_quality.joblib")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/ultra_oof.json")
    parser.add_argument("--hash-cache", type=Path, default=ROOT / "cache/train_image_sha256.joblib")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    required = {"id", "name", "description", "category", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing train columns: {missing}")
    df["name"] = df["name"].fillna("")
    df["description"] = df["description"].fillna("")

    hashes = _load_or_hash_images(df, args.images_root, args.hash_cache)
    category_results = {}
    for category in (BAD_CATEGORY, FLAMMABLE_CATEGORY):
        category_results[category] = train_category_oof(
            df, hashes, category, DEFAULT_CONFIGS[category], args.folds
        )

    text_models = {}
    for category in (BAD_CATEGORY, FLAMMABLE_CATEGORY):
        rows = df[df["category"].astype(str).eq(category)]
        print(f"Fitting final text model: {category}", flush=True)
        model = fit_text_model(
            rows["name"].tolist(),
            rows["description"].tolist(),
            rows["category"].tolist(),
            rows["label"].astype(int).tolist(),
            config=DEFAULT_CONFIGS[category],
        )
        model["threshold"] = category_results[category]["selected_policy"]["threshold"]
        model["retrieval_policy"] = {
            "mode": category_results[category]["selected_policy"]["mode"],
            "weight": category_results[category]["selected_policy"]["weight"],
        }
        text_models[category] = model

    print("Building final exact retrieval index", flush=True)
    retrieval = build_retrieval_index(
        df["name"].tolist(),
        df["description"].tolist(),
        df["category"].tolist(),
        df["label"].astype(int).tolist(),
        hashes,
    )

    artifact = {
        "format_version": ULTRA_ARTIFACT_FORMAT_VERSION,
        "feature_schema": ULTRA_FEATURE_SCHEMA,
        "trained_rows": len(df),
        "text_models": text_models,
        "retrieval": retrieval,
        "categories": [BAD_CATEGORY, FLAMMABLE_CATEGORY],
        "baseline_public_score": 0.5077170722469463,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)

    report = {
        "format_version": ULTRA_ARTIFACT_FORMAT_VERSION,
        "feature_schema": ULTRA_FEATURE_SCHEMA,
        "folds": args.folds,
        "baseline_public_score": 0.5077170722469463,
        "categories": {
            category: {
                "selected_policy": result["selected_policy"],
                "hard_retrieval_coverage": result["hard_retrieval_coverage"],
                "candidates": result["candidates"],
            }
            for category, result in category_results.items()
        },
    }
    selected_predictions = {}
    selected_f1 = {}
    for category, result in category_results.items():
        threshold = result["selected_policy"]["threshold"]
        predictions = result["selected_probabilities"] >= threshold
        selected_predictions[category] = predictions
        selected_f1[category] = f1_score(result["y"], predictions, zero_division=0)
    report["mean_f1"] = float(np.mean(list(selected_f1.values())))
    report["selected_f1"] = {key: float(value) for key, value in selected_f1.items()}

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved artifact: {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")
    print(f"Saved report: {args.report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
