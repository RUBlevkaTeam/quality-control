#!/usr/bin/env python3
"""Смесь двух каналов: TF-IDF и LoRA-VLM.

У победителей Rakuten (SIGIR-2020) сработало именно ПОЗДНЕЕ слияние -
каналы учатся отдельно, смешиваются вероятности. Раннее слияние признаков
проигрывало чистому тексту во всех их вариантах.

Прошлая наша попытка смеси дала +0.004, но вторым голосом был few-shot
с F1 0.30. Теперь у LoRA 0.84 - голос сопоставимой силы, и это другой
режим: каналы решают по-разному (счёт слов против взгляда на карточку
целиком плюс фото), поэтому их ошибки могут не совпадать.

Текстовая модель обучается на товарах, которых LoRA не видела на этом
фолде, - берём разбиение прямо из её файла предсказаний, а не пытаемся
воспроизвести тот же сид.

Запуск:
    python3 blend_lora_text.py --lora-oof qc_lora_img_labelfix/oof_fold0.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT))
else:
    sys.path.insert(0, str(ROOT.parent))

BASE = {"Легковоспламеняющиеся": 0.7986, "БАД": 0.9370}
BASE_MEAN = 0.8678


def best_f1(probs, y):
    """F1 кусочно-постоянна: перебираем только наблюдённые значения."""
    order = np.argsort(-probs)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    best = int(f1.argmax())
    return float(f1[best]), float(probs[order][best])


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-oof", required=True)
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--families", default="families.csv")
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression

    from src.retrieval_memory import absence_veto_mask, strict_positive_mask
    from src.text_model import TextQualityModel
    from src.utils_data_prep import prepare_dataframe
    from src.utils_logreg import ProductQualityPredictor

    lora = pd.read_csv(args.lora_oof)
    print(f"предсказаний LoRA: {len(lora)}")
    val_ids = set(lora["id"].tolist())

    df = prepare_dataframe(args.data, "/nonexistent")
    results = {}
    for category in ("Легковоспламеняющиеся", "БАД"):
        sub = ProductQualityPredictor._dedup(df[df.category == category]).reset_index(drop=True)
        in_val = sub["id"].isin(val_ids).values
        if in_val.sum() < 50:
            print(f"{category}: в фолде всего {int(in_val.sum())} товаров, пропуск")
            continue
        y_all = np.asarray(sub["label"].values, dtype=int)

        # обучаем текст на том, чего LoRA не видела
        train_idx = np.where(~in_val)[0]
        val_idx = np.where(in_val)[0]
        feats = TextQualityModel._make_feature_bundle(category)
        x_tr = TextQualityModel._build_category_features(category, feats, sub.iloc[train_idx], fit=True)
        x_va = TextQualityModel._build_category_features(category, feats, sub.iloc[val_idx], fit=False)
        clf = TextQualityModel._make_classifier(category, 3.0)
        clf.fit(x_tr, y_all[train_idx])
        p_text = clf.predict_proba(x_va)[:, 1]

        # те же правила, что в бою
        val_frame = sub.iloc[val_idx].reset_index(drop=True)
        if category != "БАД":
            pos_m = strict_positive_mask(val_frame)
            veto_m = absence_veto_mask(val_frame)
            p_text[pos_m] = np.maximum(p_text[pos_m], 1 - 1e-5)
            p_text[veto_m] = 1e-5

        by_id = dict(zip(lora["id"], lora["prob"]))
        p_lora = np.array([by_id[i] for i in val_frame["id"]], dtype=float)
        y = y_all[val_idx]

        f1_text, _ = best_f1(p_text, y)
        f1_lora, _ = best_f1(p_lora, y)
        print(f"\n=== {category} ===  товаров {len(y)}, позитивов {y.mean():.1%}")
        print(f"  текст соло : {f1_text:.4f}   (база на всех фолдах {BASE[category]:.4f})")
        print(f"  LoRA соло  : {f1_lora:.4f}")

        best = (f1_text, 0.0, None)
        for w in np.arange(0.05, 1.0, 0.05):
            mixed = 1 / (1 + np.exp(-((1 - w) * logit(p_text) + w * logit(p_lora))))
            f1m, thr = best_f1(mixed, y)
            if f1m > best[0]:
                best = (f1m, float(w), thr)
            if abs(w - round(w, 1)) < 1e-9 and round(w, 1) in (0.2, 0.4, 0.5, 0.6, 0.8):
                print(f"  смесь w={w:.2f} : {f1m:.4f}  ({f1m - f1_text:+.4f})")
        print(f"  ЛУЧШАЯ смесь: w={best[1]:.2f} -> {best[0]:.4f}  ({best[0] - f1_text:+.4f})")
        results[category] = (f1_text, f1_lora, best)

    if len(results) == 2:
        text_mean = np.mean([v[0] for v in results.values()])
        lora_mean = np.mean([v[1] for v in results.values()])
        mix_mean = np.mean([v[2][0] for v in results.values()])
        print(f"\n{'=' * 52}")
        print(f"среднее: текст {text_mean:.4f} | LoRA {lora_mean:.4f} | смесь {mix_mean:.4f}")
        print(f"прирост смеси над текстом: {mix_mean - text_mean:+.4f}")
        if mix_mean - text_mean >= 0.02:
            print("ВЫШЕ ПОРОГА 0.02 - есть смысл внедрять")
        else:
            print("ниже порога 0.02 - лидерборд такую разницу не различит")
        print("\nОГОВОРКА: это ОДИН фолд. Разброс F1 по fire на выборке такого")
        print("размера мы мерили как [0.686 .. 0.895] - для решения нужны все пять.")


if __name__ == "__main__":
    main()
