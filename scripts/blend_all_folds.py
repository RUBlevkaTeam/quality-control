#!/usr/bin/env python3
"""Честная оценка смеси TF-IDF + LoRA на полном OOF.

Вчерашние +0.039 получены подбором веса на том же фолде, где мерили, -
это завышает оценку. Здесь протокол строгий: вес и порог подбираются на
четырёх фолдах, применяются к пятому, и так пять раз. Итоговое число -
то, которое имеет шанс доехать до лидерборда.

Дополнительно считаем наивный вариант (подбор на всех данных) - разница
между ними и есть цена оптимизма.

Запуск:
    python3 blend_all_folds.py --lora-dir . --data data.csv --families families.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = {"Легковоспламеняющиеся": 0.7986, "БАД": 0.9370}


def best_f1(probs, y):
    order = np.argsort(-probs)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    i = int(f1.argmax())
    return float(f1[i]), float(probs[order][i])


def f1_at(probs, y, thr):
    pred = (probs >= thr).astype(int)
    tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum()); fn = int(((1 - pred) & y).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora-dir", default=".", help="где лежат qc_lora_*/oof_fold*.csv")
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--families", default="families.csv")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here if (here / "src").is_dir() else here.parent))

    from sklearn.model_selection import StratifiedGroupKFold

    from src.retrieval_memory import absence_veto_mask, strict_positive_mask
    from src.text_model import TextQualityModel
    from src.utils_data_prep import prepare_dataframe
    from src.utils_logreg import ProductQualityPredictor

    # собираем предсказания LoRA со всех фолдов
    root = Path(args.lora_dir)
    parts = sorted(root.glob("qc_lora*/oof_fold*.csv"))
    if not parts:
        raise SystemExit(f"не нашёл oof_fold*.csv в {root}/qc_lora*/")
    frames = []
    for path in parts:
        frame = pd.read_csv(path)
        frame["fold"] = int("".join(c for c in path.stem if c.isdigit()))
        frames.append(frame)
        print(f"  {path}: {len(frame)} строк")
    lora = pd.concat(frames).drop_duplicates(subset="id", keep="last")
    print(f"всего предсказаний LoRA: {len(lora)}")
    p_lora_by_id = dict(zip(lora["id"], lora["prob"]))
    fold_by_id = dict(zip(lora["id"], lora["fold"]))

    df = prepare_dataframe(args.data, "/nonexistent")
    fam = pd.read_csv(args.families)
    fam_by_id = dict(zip(fam["id"], fam["family"]))

    summary = {}
    for category in ("Легковоспламеняющиеся", "БАД"):
        sub = ProductQualityPredictor._dedup(df[df.category == category]).reset_index(drop=True)
        keep = sub["id"].isin(p_lora_by_id).values
        sub = sub[keep].reset_index(drop=True)
        y = np.asarray(sub["label"].values, dtype=int)
        groups = np.array([fam_by_id.get(i, -1) for i in sub["id"]])
        folds = np.array([fold_by_id[i] for i in sub["id"]])
        p_lora = np.array([p_lora_by_id[i] for i in sub["id"]], dtype=float)

        # OOF текстовой модели на тех же товарах, теми же family-фолдами
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

        f1_text, _ = best_f1(p_text, y)
        f1_lora, _ = best_f1(p_lora, y)
        weights = np.arange(0.05, 1.0, 0.05)

        # наивно: вес и порог подобраны на всех данных сразу
        naive = max(best_f1(1 / (1 + np.exp(-((1 - w) * logit(p_text) + w * logit(p_lora)))), y)[0]
                    for w in weights)

        # честно: подбор на четырёх фолдах, применение к пятому
        honest_preds = np.zeros(len(y), dtype=int)
        chosen = []
        for held in sorted(set(folds)):
            fit_mask = folds != held
            best = (-1.0, 0.5, 0.5)
            for w in weights:
                mixed_fit = 1 / (1 + np.exp(-((1 - w) * logit(p_text[fit_mask])
                                              + w * logit(p_lora[fit_mask]))))
                score, thr = best_f1(mixed_fit, y[fit_mask])
                if score > best[0]:
                    best = (score, float(w), thr)
            _, w_star, thr_star = best
            chosen.append((held, w_star, thr_star))
            mixed_held = 1 / (1 + np.exp(-((1 - w_star) * logit(p_text[~fit_mask])
                                           + w_star * logit(p_lora[~fit_mask]))))
            honest_preds[~fit_mask] = (mixed_held >= thr_star).astype(int)

        tp = int((honest_preds & y).sum()); fp = int((honest_preds & (1 - y)).sum())
        fn = int(((1 - honest_preds) & y).sum())
        honest = 2 * tp / max(2 * tp + fp + fn, 1e-9)

        print(f"\n=== {category} ===  товаров {len(y)}, позитивов {y.mean():.1%}")
        print(f"  текст соло        {f1_text:.4f}   (наша база {BASE[category]:.4f})")
        print(f"  LoRA соло         {f1_lora:.4f}")
        print(f"  смесь наивно      {naive:.4f}   (+{naive - f1_text:.4f})  <- завышено")
        print(f"  смесь ЧЕСТНО      {honest:.4f}   ({honest - f1_text:+.4f})  <- реалистично")
        print(f"  веса по фолдам: {[(h, round(w, 2)) for h, w, _ in chosen]}")
        summary[category] = (f1_text, f1_lora, naive, honest)

    if len(summary) == 2:
        keys = list(summary)
        mean = lambda i: np.mean([summary[k][i] for k in keys])
        print(f"\n{'=' * 56}")
        print(f"среднее по категориям:")
        print(f"  текст          {mean(0):.4f}")
        print(f"  LoRA           {mean(1):.4f}")
        print(f"  смесь наивно   {mean(2):.4f}  ({mean(2) - mean(0):+.4f})")
        print(f"  смесь ЧЕСТНО   {mean(3):.4f}  ({mean(3) - mean(0):+.4f})")
        gain = mean(3) - mean(0)
        print()
        if gain >= 0.02:
            print(f"ВНЕДРЯЕМ: честный прирост {gain:+.4f} выше порога 0.02")
        elif gain > 0:
            print(f"прирост {gain:+.4f} есть, но ниже порога 0.02 - копим с другими")
        else:
            print(f"смесь не помогает ({gain:+.4f})")
        print(f"цена оптимизма (наивно минус честно): {mean(2) - mean(3):.4f}")


if __name__ == "__main__":
    main()
