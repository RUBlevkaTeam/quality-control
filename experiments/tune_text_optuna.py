#!/usr/bin/env python3
"""Подбор гиперпараметров текстовой модели через Optuna.

Зачем: векторизаторы захардкожены (ngram, min_df, max_features, веса
правил), подбирался только C. Прогон одной конфигурации - секунды, так что
здесь Optuna уместна, в отличие от LoRA, где проба стоит часы.

Честность замера: те же family-фолды, что во всех наших числах, и НЕСКОЛЬКО
сидов - иначе оптимизатор найдёт конфигурацию, удачную для одного разбиения.
Категории подбираются отдельно: у fire 3.6% позитивов и у БАД 74.5%,
оптимумы у них разные.

Запуск:
    python3 scripts/tune_text_optuna.py --category fire --trials 150 --jobs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CATEGORIES = {"fire": "Легковоспламеняющиеся", "bad": "БАД"}


def build_features(category, params, frame, fitted=None):
    """Собираем матрицу так же, как боевой код, но с параметрами из пробы."""
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    from src.rule_features import build_model_texts, normalized_name, rule_feature_matrix

    texts = build_model_texts(frame["name"], frame["description"])
    titles = [normalized_name(v) for v in frame["name"]]
    is_fire = category == CATEGORIES["fire"]
    # у fire название повторяется несколько раз: маркер часто именно там
    if is_fire:
        texts = [f"{t} {(' ' + ttl) * params['title_repeats']}" for t, ttl in zip(texts, titles)]

    blocks, new_fitted = [], {}
    word = fitted["word"] if fitted else TfidfVectorizer(
        analyzer="word", ngram_range=(1, params["word_ngram_max"]),
        min_df=params["word_min_df"], max_features=params["word_max_features"],
        sublinear_tf=True)
    blocks.append(word.transform(texts) if fitted else word.fit_transform(texts))
    new_fitted["word"] = word

    if is_fire:
        char = fitted["char"] if fitted else TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(params["char_ngram_min"], params["char_ngram_max"]),
            min_df=params["char_min_df"], max_features=params["char_max_features"],
            sublinear_tf=True)
        blocks.append(char.transform(texts) if fitted else char.fit_transform(texts))
        new_fitted["char"] = char

    rules = rule_feature_matrix(frame["name"], frame["description"], frame["category"])
    blocks.append(sparse.csr_matrix(np.asarray(rules) * params["rule_weight"]))
    return sparse.hstack(blocks, format="csr"), new_fitted


def exact_best_f1(probs, y):
    order = np.argsort(-probs)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return float(f1.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=sorted(CATEGORIES), default="fire")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=2, help="разбиений на пробу")
    parser.add_argument("--data", default=str(ROOT / "data.csv"))
    parser.add_argument("--families", default=str(ROOT / "families.csv"))
    parser.add_argument("--storage", default="")
    args = parser.parse_args()

    import optuna
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    from src.utils_data_prep import prepare_dataframe
    from src.utils_logreg import ProductQualityPredictor

    category = CATEGORIES[args.category]
    df = prepare_dataframe(args.data, "/nonexistent")
    fam = pd.read_csv(args.families)
    fam_by_id = dict(zip(fam["id"], fam["family"]))

    sub = ProductQualityPredictor._dedup(df[df.category == category]).reset_index(drop=True)
    y = np.asarray(sub["label"].values, dtype=int)
    groups = np.array([fam_by_id.get(i, -1) for i in sub["id"]])
    print(f"{category}: {len(y)} товаров, позитивов {y.mean():.1%}", flush=True)

    is_fire = category == CATEGORIES["fire"]

    def objective(trial):
        params = {
            "word_ngram_max": trial.suggest_int("word_ngram_max", 1, 3),
            "word_min_df": trial.suggest_int("word_min_df", 1, 6),
            "word_max_features": trial.suggest_categorical(
                "word_max_features", [50_000, 100_000, 200_000, None]),
            "rule_weight": trial.suggest_float("rule_weight", 0.5, 8.0, log=True),
            "title_repeats": trial.suggest_int("title_repeats", 0, 5) if is_fire else 0,
            "char_ngram_min": trial.suggest_int("char_ngram_min", 2, 4) if is_fire else 3,
            # верхнюю границу задаём как «нижняя + ширина»: иначе половина
            # проб отсекалась бы на min >= max и уходила впустую
            "char_ngram_max": 0 if not is_fire else 6,
            "char_min_df": trial.suggest_int("char_min_df", 1, 5) if is_fire else 2,
            "char_max_features": trial.suggest_categorical(
                "char_max_features", [80_000, 160_000, 320_000]) if is_fire else 160_000,
        }
        if is_fire:
            width = trial.suggest_int("char_ngram_width", 1, 4)
            params["char_ngram_max"] = params["char_ngram_min"] + width
        else:
            params["char_ngram_max"] = 6
        C = trial.suggest_float("C", 0.05, 300.0, log=True)
        pos_weight = trial.suggest_float("pos_weight", 5.0, 400.0, log=True) if is_fire else 0.0

        scores = []
        for seed in range(42, 42 + args.seeds):
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            oof = np.zeros(len(y))
            for tr, va in cv.split(sub, y, groups):
                x_tr, fitted = build_features(category, params, sub.iloc[tr])
                x_va, _ = build_features(category, params, sub.iloc[va], fitted=fitted)
                weight = {0: 1, 1: pos_weight} if is_fire else "balanced"
                clf = LogisticRegression(C=C, max_iter=2000, class_weight=weight,
                                         solver="liblinear" if is_fire else "lbfgs",
                                         random_state=2026)
                clf.fit(x_tr, y[tr])
                oof[va] = clf.predict_proba(x_va)[:, 1]
            scores.append(exact_best_f1(oof, y))
        # минимум по сидам, а не среднее: нужна конфигурация, устойчивая
        # к разбиению, а не удачная на одном
        trial.set_user_attr("mean", float(np.mean(scores)))
        return float(np.min(scores))

    study = optuna.create_study(
        direction="maximize",
        storage=args.storage or None,
        study_name=f"text_{args.category}",
        load_if_exists=bool(args.storage),
        sampler=optuna.samplers.TPESampler(seed=2026, n_startup_trials=40, multivariate=True),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def report(study, trial):
        # у отсечённых проб value пустой - печатать нечего
        if trial.value is None:
            return
        if trial.number % 10 == 0 or trial.value == study.best_value:
            print(f"  проба {trial.number:3d}: {trial.value:.4f} "
                  f"(лучшее {study.best_value:.4f})", flush=True)

    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, callbacks=[report])

    print(f"\nлучший минимум по сидам: {study.best_value:.4f}")
    print(f"среднее у него: {study.best_trial.user_attrs.get('mean', float('nan')):.4f}")
    print("параметры:")
    for key, value in sorted(study.best_params.items()):
        print(f"    {key} = {value}")
    out = ROOT / f"optuna_text_{args.category}.json"
    out.write_text(pd.Series(study.best_params).to_json(indent=2), encoding="utf-8")
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
