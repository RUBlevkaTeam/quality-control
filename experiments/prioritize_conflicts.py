#!/usr/bin/env python3
"""Приоритизация конфликтных семейств для ручного арбитража меток.

Семейство = группа почти-дублей из families.csv. Если внутри семейства
встречаются разные label - это либо шум разметки, либо реально разные товары.
Каждая такая группа стоит ручного взгляда: чистая метка улучшает и обучение,
и retrieval-память, и OOF-оценку.

Приоритет: сначала редкий класс «Легковоспламеняющиеся» (каждый позитив
на счету), внутри категории - группы с перевесом позитивов и большим размером.

Выход: reports/conflict_queue.csv + краткая статистика в stdout.
Строки, уже разобранные в label_corrections.csv, помечаются resolved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIRE = "Легковоспламеняющиеся"


def main() -> None:
    df = pd.read_csv(ROOT / "data.csv")
    families = pd.read_csv(ROOT / "families.csv")
    corrections = pd.read_csv(ROOT / "label_corrections.csv")
    conflicts = pd.read_csv(ROOT / "conflicts_review.csv")

    df = df.merge(families, on="id", how="left")
    resolved = set(corrections["id"])

    rows = []
    for family, group in df[df["family"].notna()].groupby("family"):
        labels = set(group["label"].unique())
        if len(labels) < 2:
            continue
        n_pos = int((group["label"] == 1).sum())
        n_neg = int((group["label"] == 0).sum())
        category = str(group["category"].mode().iat[0])
        # приоритет: fire выше БАД; внутри - чем ближе счёт к равному и чем
        # больше группа, тем дороже ошибка разметки
        priority = (
            0 if category == FIRE else 1,
            -abs(n_pos - n_neg) / max(n_pos + n_neg, 1),
            -(n_pos + n_neg),
        )
        rows.append(
            {
                "priority_score": round(-priority[1] * 100 - abs(n_pos - n_neg)),
                "category": category,
                "family": int(family),
                "n_products": len(group),
                "n_label1": n_pos,
                "n_label0": n_neg,
                "ids": " ".join(map(str, sorted(group["id"]))),
                "first_name": str(group["name"].iloc[0])[:120],
                "resolved_ids": sum(int(i in resolved) for i in group["id"]),
            }
        )

    queue = pd.DataFrame(rows).sort_values(
        ["category", "n_label1", "n_products"], ascending=[False, False, False]
    )
    out = ROOT / "reports" / "conflict_queue.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(out, index=False)

    print(f"конфликтных семейств: {len(queue)}")
    for category, group in queue.groupby("category"):
        done = group[group["resolved_ids"] > 0]
        untouched = group[group["resolved_ids"] == 0]
        print(
            f"  {category}: всего {len(group)}, "
            f"с частичными правками {len(done)}, без правок {len(untouched)}"
        )
    reviewed = set(conflicts.get("family", pd.Series(dtype=float)).dropna())
    if reviewed:
        fresh = queue[~queue["family"].isin(reviewed)]
        print(f"  не отмечены в conflicts_review.csv: {len(fresh)} (см. начало файла)")
    print(f"очередь сохранена: {out}")


if __name__ == "__main__":
    main()
