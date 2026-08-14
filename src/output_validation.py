"""Строгая проверка CSV-контракта соревнования.

Result-стадия отклоняет весь файл, если хоть одна строка не соответствует
формату, поэтому дешевле упасть здесь, у себя, чем получить ноль за прогон.
Проверяется ровно то, что перечислено в условии: две колонки, ответ на
каждый входной id, формат '<комментарий>ТЕКСТ<вердикт>бан|не бан' без
закрывающих тегов и длина комментария 50..300 символов.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Sequence

MIN_COMMENT_LEN = 50
MAX_COMMENT_LEN = 300

_RESULT_RE = re.compile(
    r"^<комментарий>(?P<comment>.*?)<вердикт>(?P<verdict>бан|не бан)$",
    re.DOTALL,
)
# служебные теги внутри самого комментария сломают разбор у проверяющей системы
_RESERVED_RE = re.compile(r"</?(?:комментарий|вердикт)>", re.IGNORECASE)


def validate_result(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError(f"result должен быть строкой, а не {type(value).__name__}")
    match = _RESULT_RE.fullmatch(value)
    if not match:
        raise ValueError(f"result не соответствует формату тегов: {value[:120]!r}")
    comment = match.group("comment")
    if not MIN_COMMENT_LEN <= len(comment) <= MAX_COMMENT_LEN:
        raise ValueError(
            f"длина комментария {len(comment)}, ожидалось {MIN_COMMENT_LEN}..{MAX_COMMENT_LEN}"
        )
    # перенос строки превращает запись в многострочную и ломает наивный парсер
    if "\n" in comment or "\r" in comment:
        raise ValueError("комментарий содержит перенос строки")
    if _RESERVED_RE.search(comment):
        raise ValueError("комментарий содержит служебный тег")


def validate_rows(expected_ids: Sequence[object], rows: Iterable[tuple[object, object]]) -> None:
    materialized = list(rows)
    actual = [str(row[0]) for row in materialized]
    expected = [str(value) for value in expected_ids]
    if actual != expected:
        raise ValueError(
            f"id на выходе не совпадают со входом: строк {len(actual)} против {len(expected)}"
        )
    if len(set(actual)) != len(actual):
        raise ValueError("на выходе есть дублирующиеся id")
    for index, (_, result) in enumerate(materialized):
        try:
            validate_result(result)
        except ValueError as exc:
            raise ValueError(f"строка {index + 2}: {exc}") from exc


# читает уже записанный файл ровно так же, как это сделает проверяющая система
def validate_csv_file(path: str | Path, expected_ids: Sequence[object]) -> None:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("выходной CSV пуст") from exc
        if header != ["id", "result"]:
            raise ValueError(f"неожиданный заголовок: {header!r}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"строка {row_number}: полей {len(row)} вместо 2")
            rows.append((row[0], row[1]))
    validate_rows(expected_ids, rows)
