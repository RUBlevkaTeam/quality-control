"""Retrieval-память: перенос вердикта с train-товаров, совпавших с тестовым.

Три ключа, от жёсткого к мягкому:
  1) нормализованный name+description - точное совпадение текста;
  2) SHA-1 главной картинки - побайтовая копия файла;
  3) dHash-256 главной картинки, Хэмминг <= 6 - визуальная почти-копия.

По замерам на data.csv: 53% товаров состоят в семействах почти-дублей,
чистота меток внутри 95.8%, у редкого класса памятью покрыто 105 из 198
позитивов. Метка переносится только при единогласии совпавших train-товаров
(text/sha) либо при чистоте группы >= MIN_PURITY (dhash).

Всё на CPU: PIL + numpy, без новых зависимостей в контейнере.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MEMORY_FORMAT_VERSION = 1
DHASH_THRESH = 6        # Хэмминг для dHash-256; выверен на families (98% чистоты)
MIN_PURITY = 0.8        # минимальная доля большинства в dHash-группе
_NORM_RE = re.compile(r"[^0-9a-zа-яё]+")


def _norm_key(name: object, desc: object) -> str:
    a = "" if name is None or name != name else str(name)
    b = "" if desc is None or desc != desc else str(desc)
    key = _NORM_RE.sub("", (a + "||" + b).lower())
    return key


# BLAKE2b-128 дайджест нормализованного ОПИСАНИЯ: в артефакт едет 16 байт,
# а не сам текст. Пустое описание -> None (пустоту запоминать нельзя).
def _desc_digest(desc: object) -> bytes | None:
    text = "" if desc is None or desc != desc else str(desc)
    norm = _NORM_RE.sub("", text.lower())
    if not norm:
        return None
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).digest()


def _main_image(row) -> str | None:
    paths = row.get("image_paths")
    if isinstance(paths, (list, tuple)) and paths:
        return paths[0]
    return None


def _dhash256(path: str, size: int = 16) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
        return np.packbits(a[:, 1:] > a[:, :-1])
    except Exception:
        return None


def _sha1(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except Exception:
        return None


class RetrievalMemory:
    """fit() на train-товарах, lookup() отдаёт (label, source) или (None, "")."""

    def __init__(self) -> None:
        self.format_version = MEMORY_FORMAT_VERSION
        self.text_labels: Dict[str, int] = {}
        self.sha_labels: Dict[str, int] = {}
        self.dhash_matrix = np.zeros((0, 32), dtype=np.uint8)
        self.dhash_labels = np.zeros(0, dtype=np.int8)
        # дайджесты описаний, у которых ВСЕ train-метки = 0 (для negative lookup)
        self.neg_desc_digests: set = set()

    # --- обучение (офлайн) ---

    def fit(self, df: pd.DataFrame, *, with_images: bool = True) -> "RetrievalMemory":
        by_text: Dict[str, set] = defaultdict(set)
        by_desc: Dict[bytes, set] = defaultdict(set)
        for _, row in df.iterrows():
            key = _norm_key(row.get("name"), row.get("description"))
            if key and key != "||":
                by_text[key].add(int(row["label"]))
            digest = _desc_digest(row.get("description"))
            if digest is not None:
                by_desc[digest].add(int(row["label"]))
        # переносим метку только при единогласии: конфликтная разметка не память
        self.text_labels = {k: v.pop() for k, v in by_text.items() if len(v) == 1}
        # описания, где train единогласно говорит 0: конфликт или наличие
        # хоть одной метки 1 исключает дайджест из негативной памяти
        self.neg_desc_digests = {d for d, ls in by_desc.items() if ls == {0}}

        if with_images and "image_paths" in df.columns:
            by_sha: Dict[str, set] = defaultdict(set)
            hashes: List[np.ndarray] = []
            labels: List[int] = []
            for _, row in df.iterrows():
                p = _main_image(row)
                if p is None:
                    continue
                sha = _sha1(p)
                if sha:
                    by_sha[sha].add(int(row["label"]))
                h = _dhash256(p)
                if h is not None:
                    hashes.append(h)
                    labels.append(int(row["label"]))
            self.sha_labels = {k: v.pop() for k, v in by_sha.items() if len(v) == 1}
            if hashes:
                self.dhash_matrix = np.vstack(hashes)
                self.dhash_labels = np.asarray(labels, dtype=np.int8)
        return self

    # --- инференс ---

    # Возвращает по строке df: (метка из памяти | None, источник).
    # Порядок от жёсткого к мягкому: text -> sha -> dhash-голосование.
    def lookup(self, df: pd.DataFrame) -> Tuple[List[object], List[str]]:
        n = len(df)
        found: List[object] = [None] * n
        source: List[str] = [""] * n

        # текстовый ключ - без файловых операций, всегда доступен
        for i, (_, row) in enumerate(df.iterrows()):
            key = _norm_key(row.get("name"), row.get("description"))
            if key in self.text_labels:
                found[i] = self.text_labels[key]
                source[i] = "text"

        if "image_paths" not in df.columns:
            return found, source

        pending = [i for i in range(n) if found[i] is None]
        rows = list(df.iterrows())
        test_hashes: Dict[int, np.ndarray] = {}
        for i in pending:
            p = _main_image(rows[i][1])
            if p is None:
                continue
            sha = _sha1(p)
            if sha and sha in self.sha_labels:
                found[i] = self.sha_labels[sha]
                source[i] = "sha"
                continue
            h = _dhash256(p)
            if h is not None:
                test_hashes[i] = h

        if test_hashes and len(self.dhash_matrix):
            idx = list(test_hashes.keys())
            q = np.vstack([test_hashes[i] for i in idx])
            # XOR + аппаратный popcount: (m, n_train) расстояний за один проход
            d = np.bitwise_count(
                q[:, None, :] ^ self.dhash_matrix[None, :, :]
            ).sum(axis=2, dtype=np.uint16)
            for r, i in enumerate(idx):
                near = d[r] <= DHASH_THRESH
                support = int(near.sum())
                if support == 0:
                    continue
                votes = self.dhash_labels[near]
                share = votes.mean()
                purity = max(share, 1.0 - share)
                if purity >= MIN_PURITY:
                    found[i] = int(round(share))
                    source[i] = "dhash"
        return found, source

    # Негативный lookup по описанию: True там, где описание непустое, точно
    # найдено в train и ВСЕ его train-метки = 0. Только в сторону pred=0 -
    # редкий позитив этим убить нельзя по построению. Family-folds по 4 seed:
    # fire-F1 +0.006..0.012, исправлено/испорчено 10/0.
    def negative_desc_mask(self, df: pd.DataFrame) -> np.ndarray:
        mask = np.zeros(len(df), dtype=bool)
        if not self.neg_desc_digests:
            return mask
        for i, (_, row) in enumerate(df.iterrows()):
            digest = _desc_digest(row.get("description"))
            if digest is not None and digest in self.neg_desc_digests:
                mask[i] = True
        return mask

    def summary(self) -> str:
        return (
            f"память: текстовых ключей {len(self.text_labels)}, "
            f"sha {len(self.sha_labels)}, dhash {len(self.dhash_labels)}, "
            f"neg-desc {len(self.neg_desc_digests)}"
        )


# Absence veto для «Легковоспламеняющихся»: «газ/топливо/баллон не входит
# в комплект» и нет признака перезаправляемости -> принудительный pred=0.
# Регулярки - те же, что в rule-признаках (единый источник).
def absence_veto_mask(df: pd.DataFrame) -> np.ndarray:
    from src.rule_features import (
        _ABSENCE_VETO_RE,
        _REFILLABLE_RE,
        FLAMMABLE_CATEGORY,
        normalize_text,
    )

    mask = np.zeros(len(df), dtype=bool)
    for i, (_, row) in enumerate(df.iterrows()):
        if str(row.get("category", "")) != FLAMMABLE_CATEGORY:
            continue
        text = f"{normalize_text(row.get('name'))} {normalize_text(row.get('description'))}"
        if _ABSENCE_VETO_RE.search(text) and not _REFILLABLE_RE.search(text):
            mask[i] = True
    return mask


# Зеркальный позитивный override: высокоточное семейство fire-позитивов
# (пиротехника с ГОСТ, газ для заправки зажигалок, дымовые шашки и т.п.).
# На train: 37 срабатываний, все 37 - label=1. OOF не меняет (модель их и так
# ловит), включён как страховка от пропуска незнакомых брендов на тесте.
# Вето сильнее: если сработали оба - остаётся pred=0.
def strict_positive_mask(df: pd.DataFrame) -> np.ndarray:
    from src.rule_features import (
        _STRICT_FLAME_POSITIVE_RE,
        FLAMMABLE_CATEGORY,
        normalize_text,
    )

    mask = np.zeros(len(df), dtype=bool)
    for i, (_, row) in enumerate(df.iterrows()):
        if str(row.get("category", "")) != FLAMMABLE_CATEGORY:
            continue
        text = f"{normalize_text(row.get('name'))} {normalize_text(row.get('description'))}"
        if _STRICT_FLAME_POSITIVE_RE.search(text):
            mask[i] = True
    return mask
