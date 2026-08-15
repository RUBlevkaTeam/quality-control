"""Сборка строки результата в формате соревнования.

Все инварианты, которые проверяет output_validation, обеспечиваются здесь:
длина комментария строго 50..300 символов ПОСЛЕ strip, ни одного переноса
строки, никаких служебных тегов внутри текста.
"""

import re
from typing import List, Sequence

from src.constants import (
    MIN_COMMENT_LEN,
    MAX_COMMENT_LEN,
    MIN_COMMENT_FILLER,
    MISSING_COMMENT_PLACEHOLDER,
    VERDICT_FOR_POSITIVE,
    VERDICT_FOR_NEGATIVE,
)

_WS_RE = re.compile(r"\s+")
# служебные теги внутри комментария сломают разбор результата
_RESERVED_RE = re.compile(r"</?(?:комментарий|вердикт)>", re.IGNORECASE)


# Приводит комментарий к 50..300 символам без переносов строк.
# Возвращаемое значение уже не требует strip: хвостовых пробелов не остаётся,
# иначе валидатор, считающий длину после strip, увидел бы меньше минимума.
def _patch_comment(
    raw_comment: object,
    min_len: int = MIN_COMMENT_LEN,
    max_len: int = MAX_COMMENT_LEN,
    filler: str = MIN_COMMENT_FILLER,
    placeholder: str = MISSING_COMMENT_PLACEHOLDER,
) -> str:
    comment = "" if raw_comment is None else str(raw_comment)
    comment = _RESERVED_RE.sub(" ", comment)
    # переносы схлопываем здесь: многострочная запись ломает наивный парсер CSV
    comment = _WS_RE.sub(" ", comment).strip()

    if not comment:
        comment = placeholder

    # добиваем осмысленным текстом, а не пробелами: пробелы съест strip
    while len(comment) < min_len:
        before = len(comment)
        comment = f"{comment}{filler}".strip()
        comment = _WS_RE.sub(" ", comment)
        if len(comment) == before:  # филлер пуст, дальше расти нечему
            comment = (comment + " " + placeholder).strip()
            if len(comment) == before:
                break

    if len(comment) > max_len:
        cut = comment[:max_len]
        trim_idx = cut.rfind(" ")
        # обрезаем по слову, только если после этого остаёмся выше минимума,
        # иначе режем жёстко по символам
        comment = cut[:trim_idx].rstrip() if trim_idx >= min_len else cut.rstrip()

    return comment


def _verdict(value: object) -> str:
    if value is None:
        return VERDICT_FOR_NEGATIVE
    try:
        return VERDICT_FOR_POSITIVE if int(value) == 1 else VERDICT_FOR_NEGATIVE
    except (TypeError, ValueError):
        return VERDICT_FOR_NEGATIVE


# Собирает строки результата. Длина выхода всегда равна n (числу товаров):
# нехватка комментариев или вердиктов заполняется заглушками, а не роняет
# прогон и не даёт CSV короче входа.
def format_results(
    raw_comments: Sequence[object] | None,
    crisp_verdicts: Sequence[object] | None,
    n: int | None = None,
) -> List[str]:
    comments = list(raw_comments) if raw_comments is not None else []
    verdicts = list(crisp_verdicts) if crisp_verdicts is not None else []

    if n is None:
        n = max(len(comments), len(verdicts))
    if n == 0:
        return []

    results = []
    for i in range(n):
        comment = _patch_comment(comments[i] if i < len(comments) else "")
        verdict = _verdict(verdicts[i] if i < len(verdicts) else None)
        results.append(f"<комментарий>{comment}<вердикт>{verdict}")
    return results
