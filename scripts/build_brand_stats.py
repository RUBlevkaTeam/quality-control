#!/usr/bin/env python3
"""Строит brand_stats.json: первый токен названия -> (n товаров, n позитивов).

Хранятся только токены с n >= MIN_COUNT, отдельно по категориям, плюс приор
категории. Вложенно-честный замер (30.08): смесь текст+LoRA+бренд по fire
0.8165 против 0.7985 без бренда; конфигурация first/min3/smooth3 выбиралась
на 4 фолдах из 5 стабильно, вес канала 0.4 на всех фолдах.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brand_channel import BRAND_FORMAT_VERSION, MIN_COUNT, brand_key


def main() -> None:
    df = pd.read_csv(ROOT / "data.csv")
    categories = {}
    for category, part in df.groupby("category"):
        stats: dict = defaultdict(lambda: [0, 0])
        for name, label in zip(part["name"], part["label"]):
            key = brand_key(name)
            if key is None:
                continue
            stats[key][0] += 1
            stats[key][1] += int(label)
        kept = {k: v for k, v in stats.items() if v[0] >= MIN_COUNT}
        categories[str(category)] = {
            "prior": float(part["label"].mean()),
            "brands": kept,
        }
        covered = sum(v[0] for v in kept.values())
        print(f"{category}: брендов >= {MIN_COUNT}: {len(kept)}, "
              f"покрыто товаров {covered}/{len(part)} ({covered / len(part):.0%})")
    out = ROOT / "brand_stats.json"
    out.write_text(json.dumps(
        {"format_version": BRAND_FORMAT_VERSION, "categories": categories},
        ensure_ascii=False), encoding="utf-8")
    print(f"сохранено: {out} ({out.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
