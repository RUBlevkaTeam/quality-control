#!/usr/bin/env python3
"""Считает family-OOF вероятности ТЕКСТОВОЙ модели (сторона Colab).

Зачем: веса ансамбля w*p_text + (1-w)*p_vlm и пороги подбираются офлайн по
OOF обеих моделей на одних и тех же фолдах. Этот скрипт запускается в колабе
(обычный CPU, scikit-learn из pip), чтобы ноутбук не считал ничего.

Вход (рядом со скриптом): text_model.joblib, data.csv, families.csv,
label_corrections.csv. Выход: text_oof.csv (id, prob).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))

    from sklearn.model_selection import StratifiedGroupKFold

    from src.text_model import TextQualityModel, exact_best_threshold
    from src.utils_data_prep import prepare_dataframe
    from src.utils_logreg import ProductQualityPredictor

    df = prepare_dataframe(here / "data.csv", here / "images")
    corrections = pd.read_csv(here / "label_corrections.csv")
    fix = dict(zip(corrections["id"], corrections["corrected_label"]))
    mask = df["id"].isin(fix)
    df.loc[mask, "label"] = df.loc[mask, "id"].map(fix)

    model = TextQualityModel.load(str(here / "text_model.joblib"))
    families = pd.read_csv(here / "families.csv")
    fam_by_id = dict(zip(families["id"], families["family"]))

    oof = np.full(len(df), np.nan)
    for category in sorted(df["category"].dropna().unique()):
        subset = ProductQualityPredictor._dedup(
            df[df["category"] == category]
        ).reset_index(drop=True)
        y = subset["label"].to_numpy(dtype=np.int64)
        groups = np.asarray([fam_by_id.get(i, -1) for i in subset["id"]])

        # усредняем OOF по трём сидам - так же шумнее порог, но честнее вес бленда
        probs_sum = np.zeros(len(y))
        seeds_used = 0
        for seed in (2026, 2027, 2028):
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            info = None
            for name, stored in model.category_models.items():
                if str(name) == category:
                    info = stored
            if info is None:
                continue
            fold_probs = np.zeros(len(y))
            for train_idx, val_idx in cv.split(subset, y, groups):
                bundle = model._make_feature_bundle(category)
                x_tr = model._build_category_features(category, bundle, subset.iloc[train_idx], fit=True)
                clf = model._make_classifier(category, info["C"])
                clf.fit(x_tr, y[train_idx])
                x_va = model._build_category_features(category, bundle, subset.iloc[val_idx], fit=False)
                fold_probs[val_idx] = clf.predict_proba(x_va)[:, 1]
            f1, _ = exact_best_threshold(fold_probs, y)
            print(f"{category} seed={seed}: F1={f1:.4f}", flush=True)
            probs_sum += fold_probs
            seeds_used += 1

        if seeds_used:
            pos_of_id = {int(pid): i for i, pid in enumerate(subset["id"])}
            for i, pid in enumerate(subset["id"]):
                row_index = df.index[df["id"] == pid][0]
                oof[row_index] = probs_sum[i] / seeds_used

    out = pd.DataFrame({"id": df["id"], "prob": oof})
    out.to_csv(here / "text_oof.csv", index=False)
    print(f"готово: text_oof.csv, {out['prob'].notna().sum()} строк")


if __name__ == "__main__":
    main()
