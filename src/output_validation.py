"""Strict validation for the competition CSV contract."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Sequence


_RESULT_RE = re.compile(
    r"^<комментарий>(?P<comment>.*?)<вердикт>(?P<verdict>бан|не бан)$",
    re.DOTALL,
)
_RESERVED_RE = re.compile(r"</?(?:комментарий|вердикт)>", re.IGNORECASE)


def validate_result(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("result must be a string")
    match = _RESULT_RE.fullmatch(value)
    if not match:
        raise ValueError("result does not match the required tag format")
    comment = match.group("comment")
    if not 50 <= len(comment) <= 300:
        raise ValueError(f"comment length is {len(comment)}, expected 50..300")
    if "\n" in comment or "\r" in comment:
        raise ValueError("comment contains a newline")
    if _RESERVED_RE.search(comment):
        raise ValueError("comment contains a reserved tag")


def validate_rows(expected_ids: Sequence[object], rows: Iterable[tuple[object, object]]) -> None:
    materialized = list(rows)
    actual_ids = [row[0] for row in materialized]
    expected_as_text = [str(value) for value in expected_ids]
    actual_as_text = [str(value) for value in actual_ids]
    if actual_as_text != expected_as_text:
        raise ValueError("output ids must exactly match input ids and order")
    if len(set(actual_as_text)) != len(actual_as_text):
        raise ValueError("output contains duplicate ids")
    for _, result in materialized:
        validate_result(result)


def validate_csv_file(path: str | Path, expected_ids: Sequence[object]) -> None:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("output CSV is empty") from exc
        if header != ["id", "result"]:
            raise ValueError(f"unexpected output header: {header!r}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"row {row_number} has {len(row)} fields instead of 2")
            rows.append((row[0], row[1]))
    validate_rows(expected_ids, rows)
