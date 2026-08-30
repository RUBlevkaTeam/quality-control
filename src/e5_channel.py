"""Второй голос классификатора: плотные эмбеддинги multilingual-e5-base.

Соло этот канал заметно слабее TF-IDF (fire 0.58 против 0.80): задача про
точные маркеры («биологически активная добавка к пище», «газ не входит в
комплект»), а e5 усредняет карточку в один вектор и формулировку теряет.
Но ошибается он В ДРУГИХ местах, поэтому логит-смесь сильнее любого из двух.

Замеры на family-фолдах, 4 сида, категория «Легковоспламеняющиеся»:
    база (TF-IDF+правила)      0.7986
    + e5-small (0.47 ГБ)       0.7976  (-0.0010, шум)
    + e5-base  (1.11 ГБ)       0.8046  (+0.0061)
    + e5-large (2.24 ГБ)       0.8064  (+0.0078)
base берёт 78% выигрыша large за половину веса - при лимите архива 5 ГБ это
единственный разумный размер. small слишком груб, чтобы что-то добавить.

Считается голыми transformers: sentence-transformers в базовом образе нет,
а результат при этом побитово совпадает (проверено, max|Δ| = 0.0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# e5 обучалась в режиме query/passage - без префикса качество падает
_PREFIX = "passage: "
_MAX_CHARS = 1500
_MAX_TOKENS = 512
E5_EMBEDDING_DIM = 768


def _log(message: str) -> None:
    print(f"[e5] {message}", file=sys.stderr, flush=True)


def build_e5_texts(df: pd.DataFrame) -> list[str]:
    """Тот же текст, что при обучении канала. Держим здесь, чтобы train и
    inference не разъехались."""
    from src.utils_data_prep import _clean

    name = _clean(df["name"]) if "name" in df.columns else pd.Series([""] * len(df))
    desc = _clean(df["description"]) if "description" in df.columns else pd.Series([""] * len(df))
    name = name.reset_index(drop=True)
    desc = desc.reset_index(drop=True)
    return (_PREFIX + name + ". " + desc).str.slice(0, _MAX_CHARS).tolist()


def encode(texts: list[str], model_path: str | Path, batch_size: int = 64) -> np.ndarray:
    """Тексты -> L2-нормированные эмбеддинги (n, 768). Mean pooling по маске."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from src.device import free_memory, select_device

    if not texts:
        return np.zeros((0, E5_EMBEDDING_DIM), dtype=np.float32)

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).eval().to(device)

    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size],
                padding=True, truncation=True,
                max_length=_MAX_TOKENS, return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.float().cpu().numpy())

    del model, tokenizer
    free_memory(device)
    matrix = np.vstack(chunks).astype(np.float32)

    # неконечные значения дальше уронили бы predict_proba
    if not np.isfinite(matrix).all():
        bad = ~np.isfinite(matrix).all(axis=1)
        _log(f"{int(bad.sum())} эмбеддингов содержат NaN/inf, обнулены")
        matrix[bad] = 0.0
    return matrix


# логит-смесь: веса подобраны замером, 0.5/0.5 не уступало другим вариантам
def blend_logits(base_probs: np.ndarray, e5_probs: np.ndarray, weight: float = 0.5) -> np.ndarray:
    def logit(p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    mixed = (1 - weight) * logit(base_probs) + weight * logit(e5_probs)
    return 1.0 / (1.0 + np.exp(-mixed))
