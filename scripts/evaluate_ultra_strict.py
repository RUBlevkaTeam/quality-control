#!/usr/bin/env python3
"""Exact-text-grouped evaluation of the ultra text/rule fallback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_ultra import DEFAULT_CONFIGS
from src.ultra_features import (
    BAD_CATEGORY,
    FLAMMABLE_CATEGORY,
    evidence_flags,
    normalized_full_text,
)
from src.ultra_model import (
    apply_empirical_absence_veto,
    best_f1_threshold,
    fit_text_model,
    predict_text_probabilities,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=ROOT / "content/content_ecup/data.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ultra_strict_cv.json")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    df["name"] = df["name"].fillna("")
    df["description"] = df["description"].fillna("")
    results = {}

    for category in (BAD_CATEGORY, FLAMMABLE_CATEGORY):
        rows = df[df["category"].astype(str).eq(category)].reset_index(drop=True)
        y = rows["label"].astype(int).to_numpy()
        groups = np.asarray(
            [
                normalized_full_text(name, description)
                for name, description in zip(rows["name"], rows["description"])
            ],
            dtype=object,
        )
        splitter = StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=2026
        )
        oof = np.zeros(len(rows), dtype=np.float64)
        fold_metrics = []
        splits = list(splitter.split(rows, y, groups))
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            train = rows.iloc[train_idx]
            val = rows.iloc[val_idx]
            model = fit_text_model(
                train["name"].tolist(),
                train["description"].tolist(),
                train["category"].tolist(),
                train["label"].astype(int).tolist(),
                config=DEFAULT_CONFIGS[category],
            )
            oof[val_idx] = predict_text_probabilities(
                model,
                val["name"].tolist(),
                val["description"].tolist(),
                val["category"].tolist(),
            )
            print(f"{category}: exact-text fold {fold}/{args.folds}", flush=True)

        flag_rows = [
            evidence_flags(row["name"], row["description"])
            for _, row in rows.iterrows()
        ]
        oof, _ = apply_empirical_absence_veto(
            oof,
            rows["category"].tolist(),
            flag_rows,
        )

        threshold, _ = best_f1_threshold(y, oof)
        pred = oof >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        for train_idx, val_idx in splits:
            fold_y = y[val_idx]
            fold_pred = oof[val_idx] >= threshold
            p, r, score, _ = precision_recall_fscore_support(
                fold_y, fold_pred, average="binary", zero_division=0
            )
            fold_metrics.append(
                {"precision": float(p), "recall": float(r), "f1": float(score)}
            )
        results[category] = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "folds": fold_metrics,
        }
        print(json.dumps({category: results[category]}, ensure_ascii=False, indent=2))

    payload = {
        "protocol": "StratifiedGroupKFold over exact normalized name+description",
        "n_splits": args.folds,
        "categories": results,
        "mean_f1": float(np.mean([entry["f1"] for entry in results.values()])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
