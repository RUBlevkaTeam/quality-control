#!/usr/bin/env python3
"""Итоговые вес и порог для боевого ансамбля -> lora_blend.json.

Вес берём медианный по фолдам (он оказался устойчивым: fire 0.4-0.6,
БАД 0.3-0.4), а порог при этом весе ищем на всём OOF - это единственный
параметр, который потом применяется к неизвестным данным.

Порог берём как СЕРЕДИНУ ПЛАТО максимума, а не первую точку: F1 кусочно
постоянна, и на плато все пороги равноценны на валидации, но середина
дальше от края, где значение обрывается.

Запуск:
    python3 make_blend_config.py --lora-dir . --data data.csv --families families.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def f1_curve(probs, y, grid):
    out = []
    for t in grid:
        pred = (probs >= t).astype(int)
        tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum())
        fn = int(((1 - pred) & y).sum())
        out.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora-dir", default=".")
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--families", default="families.csv")
    ap.add_argument("--out", default="lora_blend.json")
    ap.add_argument("--weight-fire", type=float, default=0.5)
    ap.add_argument("--weight-bad", type=float, default=0.35)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here if (here / "src").is_dir() else here.parent))

    from sklearn.model_selection import StratifiedGroupKFold

    from src.retrieval_memory import absence_veto_mask, strict_positive_mask
    from src.text_model import TextQualityModel
    from src.utils_data_prep import prepare_dataframe
    from src.utils_logreg import ProductQualityPredictor

    parts = sorted(Path(args.lora_dir).glob("qc_lora*/oof_fold*.csv"))
    lora = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset="id", keep="last")
    by_id = dict(zip(lora["id"], lora["prob"]))
    print(f"предсказаний LoRA: {len(by_id)}")

    df = prepare_dataframe(args.data, "/nonexistent")
    fam = pd.read_csv(args.families)
    fam_by_id = dict(zip(fam["id"], fam["family"]))

    weights = {"Легковоспламеняющиеся": args.weight_fire, "БАД": args.weight_bad}
    config = {"weights": {}, "thresholds": {}}
    grid = np.linspace(0.005, 0.995, 991)

    for category, w in weights.items():
        sub = ProductQualityPredictor._dedup(df[df.category == category]).reset_index(drop=True)
        sub = sub[sub["id"].isin(by_id)].reset_index(drop=True)
        y = np.asarray(sub["label"].values, dtype=int)
        groups = np.array([fam_by_id.get(i, -1) for i in sub["id"]])

        p_text = np.zeros(len(y))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        for tr, va in cv.split(sub, y, groups):
            feats = TextQualityModel._make_feature_bundle(category)
            x_tr = TextQualityModel._build_category_features(category, feats, sub.iloc[tr], fit=True)
            x_va = TextQualityModel._build_category_features(category, feats, sub.iloc[va], fit=False)
            clf = TextQualityModel._make_classifier(category, 3.0)
            clf.fit(x_tr, y[tr])
            p_text[va] = clf.predict_proba(x_va)[:, 1]
        if category != "БАД":
            pos_m = strict_positive_mask(sub); veto_m = absence_veto_mask(sub)
            p_text[pos_m] = np.maximum(p_text[pos_m], 1 - 1e-5)
            p_text[veto_m] = 1e-5

        p_lora = np.array([by_id[i] for i in sub["id"]], dtype=float)
        mixed = 1 / (1 + np.exp(-((1 - w) * logit(p_text) + w * logit(p_lora))))

        scores = f1_curve(mixed, y, grid)
        peak = scores.max()
        plateau = grid[scores >= peak - 1e-12]
        thr = float(np.median(plateau))
        key = category.strip().lower()
        config["weights"][key] = float(w)
        config["thresholds"][key] = thr
        print(f"\n{category}: вес LoRA {w:.2f}")
        print(f"  F1 на OOF {peak:.4f}, порог {thr:.3f} "
              f"(плато {plateau.min():.3f}..{plateau.max():.3f}, ширина {len(plateau)})")
        # насколько F1 просядет, если истинный оптимум сдвинется
        for delta in (0.05, 0.10):
            near = scores[(grid >= thr - delta) & (grid <= thr + delta)]
            print(f"  при сдвиге порога на ±{delta:.2f}: F1 не ниже {near.min():.4f}")

    Path(args.out).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {args.out}")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
