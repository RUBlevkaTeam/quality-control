#!/usr/bin/env python3
"""Эмбеддинги текстов train (Qwen3-VL-Embedding-2B) для kNN-переноса меток.

Гипотеза 30.08: тест может быть почти-дублями train-карточек (пережатые
картинки, слегка переписанные описания) — точные ключи (шаблон-220, SHA)
их не видят, эмбеддинги видят. Сохраняет npz: ids, emb (fp16, L2-норм),
labels, categories.

Запуск на сервере (~/ecup):
    python3 embed_train_texts.py --model /home/jupyter/qwen_embed --out train_text_emb.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT if (ROOT / "src").is_dir() else ROOT.parent))

from src.utils_data_prep import prepare_dataframe
from src.utils_embed_cuda import embed_data_cuda


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="train_text_emb.npz")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    df = prepare_dataframe(args.data, "/nonexistent")  # без картинок: чистый текст
    raw = pd.read_csv(args.data)[["id", "label"]].drop_duplicates("id")
    label_by_id = dict(zip(raw["id"], raw["label"]))

    emb = embed_data_cuda(args.model, df, batch_size=args.batch_size)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.maximum(norms, 1e-9)).astype(np.float16)

    np.savez_compressed(
        args.out,
        ids=df["id"].to_numpy(),
        emb=emb,
        labels=np.array([int(label_by_id.get(i, -1)) for i in df["id"]], dtype=np.int8),
        categories=df["category"].astype(str).to_numpy(),
    )
    print(f"сохранено: {args.out} ({Path(args.out).stat().st_size / 1e6:.0f} МБ), "
          f"товаров {len(df)}, dim {emb.shape[1]}")


if __name__ == "__main__":
    main()
