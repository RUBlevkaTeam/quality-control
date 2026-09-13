"""Бренд-канал: статистика политики разметки по первому токену названия.

Открытие 30.08: единственный канал, который не поглощается LoRA-смесью
(вложенно-честные +0.018 fire, вес 0.4 стабилен на всех фолдах). Первый
токен названия несёт бренд либо тип товара; их доля позитивов в train -
сигнал уровня «кого шерстили разметчики», переносится на незнакомые
карточки (в отличие от шаблонного переноса).

Артефакт brand_stats.json строит scripts/build_brand_stats.py по полному
train. На инференсе: сглаженная доля позитивов бренда, для непокрытых -
приор категории (так же считалась валидация).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MIN_TOKEN_LEN = 3
MIN_COUNT = 3
SMOOTH = 3.0
BRAND_FORMAT_VERSION = 1


def _log(message: str) -> None:
    print(f"[brand] {message}", file=sys.stderr, flush=True)


def brand_key(name: object) -> str | None:
    from src.rule_features import normalize_text

    tokens = normalize_text(name).split()
    if tokens and len(tokens[0]) >= MIN_TOKEN_LEN:
        return tokens[0]
    return None


def load_stats(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format_version") != BRAND_FORMAT_VERSION:
        raise ValueError(f"неожиданная версия brand_stats: {data.get('format_version')}")
    return data


def brand_prob(stats: dict, category: object, name: object) -> float:
    """Сглаженная доля позитивов бренда; приор категории, если бренд не покрыт."""
    table = stats.get("categories", {}).get(str(category))
    if not table:
        return 0.5
    prior = float(table.get("prior", 0.5))
    key = brand_key(name)
    if key is None:
        return prior
    entry = table.get("brands", {}).get(key)
    if not entry:
        return prior
    count, positives = float(entry[0]), float(entry[1])
    if count < MIN_COUNT:
        return prior
    return (positives + SMOOTH * prior) / (count + SMOOTH)
