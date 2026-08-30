"""Перенос меток по шаблонам описаний продавца.

Открытие 28.08: метка привязана к карточке/шаблону продавца, а не к типу
товара. Шаблон = первые 220 нормализованных символов описания. Внутри train
перенос по чистым шаблонам (группа >=3, чистота >=0.9) даёт точность
99.9% на fire и 98.6% на БАД, на позитивных решениях 99%.

Покрытие теста заранее неизвестно (логи сабмита не видны): если тест не
делит шаблоны с train - слой молчит и ничего не меняет; если делит -
каждый накрытый позитив редкого класса почти бесплатен.

Слой применяется ПОСЛЕ смеси и ДО image_retrieval: точные картиночные
совпадения (99-100%) сохраняют последнее слово.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MIN_PREFIX_CHARS = 40      # куцые описания шаблоном не считаем
PREFIX_CHARS = 220
NAME_PREFIX_CHARS = 60     # второй ключ: нормализованное название
MIN_NAME_CHARS = 15        # короткие названия («крем», «уголь») не ключ
MIN_GROUP = 2              # LOO-сетка 28.08: ослабление 3->2 не роняет
                           # точность (для пары вердикт = единогласие),
                           # покрытие позитивов fire растёт 84->114 из 198
MIN_PURITY = 0.9
TRANSFER_FORMAT_VERSION = 2


def _log(message: str) -> None:
    print(f"[template] {message}", file=sys.stderr, flush=True)


def template_key(description: object) -> str | None:
    from src.rule_features import normalize_text

    prefix = normalize_text(description)[:PREFIX_CHARS]
    if len(prefix) < MIN_PREFIX_CHARS:
        return None
    return hashlib.blake2b(prefix.encode("utf-8"), digest_size=8).hexdigest()


def name_key(name: object) -> str | None:
    """Второй ключ переноса. На train (LOO) даёт 0-1 ошибку на категорию и
    ни одного конфликта с ключом-описанием; описание всегда приоритетнее."""
    from src.rule_features import normalize_text

    prefix = normalize_text(name)[:NAME_PREFIX_CHARS]
    if len(prefix) < MIN_NAME_CHARS:
        return None
    return hashlib.blake2b(("n:" + prefix).encode("utf-8"), digest_size=8).hexdigest()


def load_index(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format_version") != TRANSFER_FORMAT_VERSION:
        raise ValueError(f"неожиданная версия индекса: {data.get('format_version')}")
    return data


def _verdict(table: dict, key: str | None):
    """(вердикт, размер группы) или None, если ключ не решает."""
    if key is None:
        return None
    stats = table.get(key)
    if not stats:
        return None
    count, positives = int(stats[0]), int(stats[1])
    if count < MIN_GROUP:
        return None
    share = positives / count
    if share >= MIN_PURITY:
        return 1, count
    if share <= 1.0 - MIN_PURITY:
        return 0, count
    return None


def apply_transfer(dataframe, probabilities, predictions, index: dict):
    """Возвращает (probs, preds, reasons); reasons != 'model' там, где сработало.

    Два ключа: описание (приоритетно) и название. На train-LOO конфликтов
    между ними нет ни одного; название добавляет покрытие там, где
    описание куцее или уникальное.
    """
    import numpy as np

    probs = np.asarray(probabilities, dtype=float).copy()
    preds = np.asarray(predictions, dtype=int).copy()
    reasons = np.asarray(["model"] * len(dataframe), dtype=object)
    tables = index.get("categories", {})
    changed = 0
    for position, (_, row) in enumerate(dataframe.iterrows()):
        table = tables.get(str(row.get("category", "")))
        if not table:
            continue
        hit = _verdict(table.get("desc", {}), template_key(row.get("description")))
        source = "template"
        if hit is None:
            hit = _verdict(table.get("name", {}), name_key(row.get("name")))
            source = "tname"
        if hit is None:
            continue
        verdict, count = hit
        if verdict != preds[position]:
            changed += 1
        preds[position] = verdict
        probs[position] = 1.0 - 1e-5 if verdict else 1e-5
        reasons[position] = f"{source}_{'pos' if verdict else 'neg'}_{count}"
    _log(f"перенос по шаблонам: совпало {int((reasons != 'model').sum())}, "
         f"вердикт изменён у {changed}")
    return probs, preds, reasons
