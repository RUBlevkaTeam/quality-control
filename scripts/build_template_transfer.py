#!/usr/bin/env python3
"""Строит template_transfer.json: шаблон описания -> (n товаров, n позитивов).

Хранятся ТОЛЬКО группы, пригодные для переноса (n >= MIN_GROUP), отдельно
по категориям. Точность переноса измерена leave-one-out: fire 99.9%,
БАД 98.6%, позитивные решения 99%.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.template_transfer import MIN_GROUP, TRANSFER_FORMAT_VERSION, name_key, template_key


def _build_table(values, labels, keyfn) -> dict:
    stats: dict = defaultdict(lambda: [0, 0])
    for value, label in zip(values, labels):
        key = keyfn(value)
        if key is None:
            continue
        stats[key][0] += 1
        stats[key][1] += int(label)
    return {k: v for k, v in stats.items() if v[0] >= MIN_GROUP}


def main() -> None:
    df = pd.read_csv(ROOT / "data.csv")
    categories = {}
    for category, part in df.groupby("category"):
        tables = {
            "desc": _build_table(part["description"], part["label"], template_key),
            "name": _build_table(part["name"], part["label"], name_key),
        }
        categories[str(category)] = tables
        for tag, kept in tables.items():
            pure_pos = sum(1 for v in kept.values() if v[1] / v[0] >= 0.9)
            pure_neg = sum(1 for v in kept.values() if v[1] / v[0] <= 0.1)
            print(f"{category} [{tag}]: групп >= {MIN_GROUP}: {len(kept)} "
                  f"(чисто-позитивных {pure_pos}, чисто-негативных {pure_neg})")
    out = ROOT / "template_transfer.json"
    out.write_text(json.dumps(
        {"format_version": TRANSFER_FORMAT_VERSION, "categories": categories},
        ensure_ascii=False), encoding="utf-8")
    print(f"сохранено: {out} ({out.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
