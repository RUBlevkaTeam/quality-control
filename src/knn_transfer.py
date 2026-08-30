"""kNN-перенос меток по эмбеддингам текста (Qwen3-VL-Embedding-2B).

Ставка 30.08: тест может быть почти-дублями train-карточек — пережатые
картинки и слегка переписанные описания, которые точные ключи (шаблон-220,
SHA картинок) не ловят, а эмбеддинги ловят. Слой применяется ПОСЛЕ
шаблонного переноса и ДО image retrieval; если тест почти-дублей не делит —
молчит (гейт по косинусу калиброван на train LOO между строгими группами).

Артефакты: knn_transfer.npz (ids/emb/labels/categories train) и
knn_transfer.json (порог косинуса, k, минимум согласных соседей).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KNN_FORMAT_VERSION = 1


def _log(message: str) -> None:
    print(f"[knn] {message}", file=sys.stderr, flush=True)


def load_index(npz_path: str | Path, config_path: str | Path):
    import numpy as np

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("format_version") != KNN_FORMAT_VERSION:
        raise ValueError(f"неожиданная версия knn-конфига: {config.get('format_version')}")
    data = np.load(npz_path, allow_pickle=False)
    return {
        "emb": data["emb"].astype(np.float32),
        "labels": data["labels"].astype(int),
        "categories": data["categories"],
        "config": config,
    }


def apply_knn_transfer(dataframe, probabilities, predictions, index,
                       embed_model_path: str):
    """(probs, preds, reasons); reasons != 'model' там, где сработал перенос."""
    import numpy as np

    from src.utils_embed_cuda import embed_data_cuda

    probs = np.asarray(probabilities, dtype=float).copy()
    preds = np.asarray(predictions, dtype=int).copy()
    reasons = np.asarray(["model"] * len(dataframe), dtype=object)
    per_cat = index["config"].get("categories", {})

    frame = dataframe.copy().reset_index(drop=True)
    frame["image_paths"] = [[] for _ in range(len(frame))]  # только текст
    all_categories = frame["category"].astype(str).to_numpy()
    active = np.isin(all_categories, list(per_cat))
    if not active.any():
        _log("нет включённых категорий - слой выключен")
        return probs, preds, reasons
    # эмбеддим только включённые категории (экономия времени контейнера)
    sub = frame[active].reset_index(drop=True)
    sub_emb = embed_data_cuda(embed_model_path, sub)
    norms = np.linalg.norm(sub_emb, axis=1, keepdims=True)
    sub_emb = (sub_emb / np.maximum(norms, 1e-9)).astype(np.float32)
    test_emb = np.zeros((len(frame), sub_emb.shape[1]), dtype=np.float32)
    test_emb[np.where(active)[0]] = sub_emb

    changed = matched = 0
    test_categories = all_categories
    for category in np.unique(test_categories):
        gate = per_cat.get(str(category))
        if not gate:
            continue  # категория не включена (БАД: точность LOO ниже модели)
        cos_thr = float(gate["cos_thr"])
        top_k = int(gate.get("k", 5))
        min_agree = int(gate.get("min_agree", 1))
        train_mask = index["categories"] == category
        if not train_mask.any():
            continue
        base = index["emb"][train_mask]
        base_labels = index["labels"][train_mask]
        rows = np.where(test_categories == category)[0]
        sims = test_emb[rows] @ base.T  # (n_test_cat, n_train_cat)
        for local, row in enumerate(rows):
            sim = sims[local]
            order = np.argsort(-sim)[:top_k]
            close = order[sim[order] >= cos_thr]
            if len(close) < min_agree:
                continue
            labels = base_labels[close]
            if labels.min() != labels.max():
                continue  # соседи не единогласны
            verdict = int(labels[0])
            matched += 1
            if verdict != preds[row]:
                changed += 1
            preds[row] = verdict
            probs[row] = 1.0 - 1e-5 if verdict else 1e-5
            reasons[row] = f"knn_{'pos' if verdict else 'neg'}_{len(close)}"
    _log(f"kNN-перенос: совпало {matched}, вердикт изменён у {changed}")
    return probs, preds, reasons
