#!/usr/bin/env python3
"""Офлайн-замер трёх способов снижения дисперсии поверх текстовой модели.

Варианты на одинаковых family-grouped фолдах (тот же протокол, что в
text_model.train, включая дедуп):
  base    - текущий пайплайн (basic/rich фичи + rule-признаки);
  bag     - K логрегов на бутстрап-выборках train-матрицы, среднее prob;
  smooth  - базовые prob, усреднённые внутри тестовых семейств
            (нормализованный name+desc + SHA главной картинки);
  struct  - базовые фичи + структурные признаки сырой карточки;
  bag+smooth, struct+smooth, full - комбинации.

Порог для каждого варианта подбирается exact_best_threshold на его же OOF,
как в бою. Печатает F1 по категориям и среднему по сидам.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy import sparse

from src.rule_features import FLAMMABLE_CATEGORY, rule_feature_matrix
from src.text_model import (
    TextQualityModel,
    _build_basic_features,
    _build_rich_features,
    exact_best_threshold,
)
from src.utils_data_prep import prepare_dataframe
from src.utils_logreg import ProductQualityPredictor, _norm_category

SEEDS = (2026, 2027, 2028)
N_SPLITS = 5
BAG_K = 8
CANDIDATES = ("base", "bag", "smooth", "struct", "bag_smooth", "struct_smooth", "full")

# --- структурные признаки сырой карточки -------------------------------

_TAG_COUNT_RE = re.compile(r"<[^>]*>")
_DOSAGE_RE = re.compile(r"\d+\s*(?:мг|мкг|мл|ме|iu)\b", re.IGNORECASE)
_SECTION_RE = re.compile(
    r"(?:состав|противопоказани\w*|способ\s+применени\w*|особые\s+указани\w*|"
    r"условия\s+хранени\w*|показания\s+к\s+применению)"
)


def structural_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    name_raw = df["name"].fillna("").astype(str) if "name" in df.columns else pd.Series([""] * len(df))
    desc_raw = (
        df["description"].fillna("").astype(str)
        if "description" in df.columns
        else pd.Series([""] * len(df))
    )
    rows = np.empty((len(df), 7), dtype=np.float32)
    for i, (name, desc) in enumerate(zip(name_raw, desc_raw)):
        tags = len(_TAG_COUNT_RE.findall(desc))
        dosages = len(_DOSAGE_RE.findall(desc))
        sections = len(set(_SECTION_RE.findall(desc.lower())))
        rows[i] = (
            np.log1p(tags),
            float(sections >= 3),
            float(dosages > 0),
            min(dosages, 10.0),
            np.log1p(len(desc)),
            np.log1p(len(name)),
            float(bool(desc.strip())),
        )
    return rows


def family_keys(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Ключи тестовых семейств: нормализованный name+desc и SHA главной картинки."""
    norm = (
        df["name"].fillna("").astype(str).str.lower()
        + "||"
        + df["description"].fillna("").astype(str).str.lower()
    ).str.replace(r"[^0-9a-zа-яё]+", "", regex=True)

    import hashlib

    def main_sha(paths) -> str:
        try:
            if isinstance(paths, (list, tuple)) and paths:
                with open(paths[0], "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            pass
        return ""

    sha = df["image_paths"].map(main_sha) if "image_paths" in df.columns else pd.Series([""] * len(df))
    return norm.to_numpy(), sha.to_numpy()


def group_means(probs: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Среднее prob внутри каждой группы ключа; синглтоны остаются как есть."""
    smoothed = probs.copy()
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.r_[sorted_keys[:-1] != sorted_keys[1:], True]) + 1
    starts = np.r_[0, boundaries[:-1]]
    for start, end in zip(starts, boundaries):
        if end - start < 2:
            continue
        smoothed[order[start:end]] = probs[order[start:end]].mean()
    return smoothed


def make_features(mode: str, bundle: dict, df: pd.DataFrame, fit: bool, use_struct: bool):
    if mode == "rich":
        matrix = _build_rich_features(bundle, df, fit)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = bundle.get("word")
        if vectorizer is None:
            vectorizer = bundle["word"] = TextQualityModel._make_basic_vectorizer()
        matrix = _build_basic_features(vectorizer, df, fit)
    if use_struct:
        struct = structural_feature_matrix(df) * np.float32(2.0)
        matrix = sparse.hstack((matrix, sparse.csr_matrix(struct)), format="csr", dtype=np.float32)
    return matrix


def oof_probs(category: str, subset: pd.DataFrame, y: np.ndarray, folds, *, use_struct: bool, bag: bool):
    mode = "rich" if str(category) == FLAMMABLE_CATEGORY else "basic"
    n = len(y)
    base = np.zeros(n)
    bagged = np.zeros(n) if bag else None
    rule_weight = 3.0 if mode == "rich" else 2.0
    for tr, va in folds:
        bundle = (
            TextQualityModel._make_rich_vectorizers()
            if mode == "rich"
            else {}
        )
        x_tr = make_features(mode, bundle, subset.iloc[tr], True, use_struct)
        x_va = make_features(mode, bundle, subset.iloc[va], False, use_struct)
        clf = TextQualityModel._make_classifier(category, 1.0 if mode == "basic" else 1.0)
        clf.fit(x_tr, y[tr])
        p = clf.predict_proba(x_va)[:, 1]
        base[va] = p
        if bag:
            rng = np.random.default_rng(90210 + int(y[tr].sum()))
            acc = np.zeros_like(p)
            for _ in range(BAG_K):
                idx = rng.integers(0, len(tr), len(tr))
                part = TextQualityModel._make_classifier(category, 1.0)
                part.fit(x_tr[idx], y[tr][idx])
                acc += part.predict_proba(x_va)[:, 1]
            bagged[va] = acc / BAG_K
    return base, bagged


def main() -> None:
    t0 = time.time()
    df = prepare_dataframe(ROOT / "data.csv", ROOT / "images")
    fam_by_id = dict(zip(*[pd.read_csv(ROOT / "families.csv")[c] for c in ("id", "family")]))
    corrections = pd.read_csv(ROOT / "label_corrections.csv")
    fix = dict(zip(corrections["id"], corrections["corrected_label"]))
    mask = df["id"].isin(fix)
    df.loc[mask, "label"] = df.loc[mask, "id"].map(fix)
    print(f"данные {len(df)}, правок {int(mask.sum())}", flush=True)

    text_keys, sha_keys = family_keys(df)

    from sklearn.model_selection import StratifiedGroupKFold

    per_cat: dict[str, dict[str, list[float]]] = {}

    for category in sorted(df["category"].dropna().unique()):
        subset = ProductQualityPredictor._dedup(df[df["category"] == category]).reset_index(drop=True)
        y = subset["label"].to_numpy(dtype=np.int64)
        groups = np.asarray([fam_by_id.get(i, -1) for i in subset["id"]])
        pos_in_fam = int(((y == 1) & (groups != -1)).sum())
        print(f"\n=== {category}: n={len(y)}, позитивов {y.sum()} "
              f"(в семействах {pos_in_fam}) ===", flush=True)

        scores: dict[str, list[float]] = {name: [] for name in CANDIDATES}
        key_norm = []
        key_sha = []
        # маппинг строк subset -> строки df для ключей
        pos_of_id = pd.Series(np.arange(len(df)), index=df["id"])
        sub_pos = pos_of_id.loc[subset["id"]].to_numpy()

        for seed in SEEDS:
            cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
            folds = list(cv.split(subset, y, groups))
            base, bagged = oof_probs(category, subset, y, folds, use_struct=False, bag=True)
            _, base_struct = None, None
            # struct считаем отдельным прогоном только если нужно
            struct_base, struct_bagged = oof_probs(category, subset, y, folds, use_struct=True, bag=True)

            tk = text_keys[sub_pos]
            sk = sha_keys[sub_pos]

            variants = {
                "base": base,
                "bag": bagged,
                "smooth": group_means(base, tk),
                "struct": struct_base,
                "bag_smooth": group_means(bagged, tk),
                "struct_smooth": group_means(struct_base, tk),
                "full": group_means(struct_bagged, tk),
            }
            row = f"seed={seed}: "
            for name, probs in variants.items():
                f1, thr = exact_best_threshold(probs, y)
                scores[name].append(f1)
                row += f"{name}={f1:.4f}(t{thr:.2f}) "
            print(row, flush=True)
            del struct_base, struct_bagged

        print(f"--- {category} среднее по {len(SEEDS)} сидам:")
        summary = {}
        for name in CANDIDATES:
            mean = float(np.mean(scores[name]))
            std = float(np.std(scores[name]))
            summary[name] = mean
            per_cat.setdefault(category, {})[name] = scores[name]
            print(f"  {name:>14}: {mean:.4f} ±{std:.4f}")

    print("\n===== СРЕДНЕЕ ПО КАТЕГОРИЯМ =====")
    for name in CANDIDATES:
        cats = [np.mean(v[name]) for v in per_cat.values()]
        print(f"  {name:>14}: {np.mean(cats):.4f}")
    print(f"\nвремя: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
