#!/usr/bin/env python3
"""Сборка маленького мета-пака для Colab/Kaggle (всё, кроме картинок).

Внутри qc_pack.zip:
  train.jsonl            - id, name, description, category, label, family,
                           image_paths (относительные пути вида images/<id>/<n>.jpg);
  families.csv, label_corrections.csv - как в репо;
  dump_text_oof.py       - скрипт подсчёта OOF текстовой модели на стороне колаба;
  RUN.md                 - шпаргалка по запуску.

Картинки НЕ пакуются: 9 ГБ заливаются отдельно один раз (Kaggle Dataset или
Google Drive), ноутбук при этом не считается вообще.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils_data_prep import prepare_dataframe


def main() -> None:
    df = prepare_dataframe(ROOT / "data.csv", ROOT / "images")
    corrections = pd_read_corrections()

    records = []
    for _, row in df.iterrows():
        label = int(row["label"])
        product_id = int(row["id"])
        if product_id in corrections:
            label = corrections[product_id]
        rel_paths = [
            str(Path(p).relative_to(ROOT)).replace("\\", "/")
            for p in row["image_paths"]
        ]
        records.append(
            {
                "id": product_id,
                "name": str(row["name"]),
                "description": str(row["description"]),
                "category": str(row["category"]),
                "label": label,
                "image_paths": rel_paths,
            }
        )

    out_dir = ROOT / "qc_pack"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    for name in ("families.csv", "label_corrections.csv", "data.csv"):
        source = ROOT / name
        if source.exists():
            (out_dir / name).write_bytes(source.read_bytes())

    for extra in ("dump_text_oof.py",):
        source = ROOT / "scripts" / extra
        if source.exists():
            (out_dir / extra).write_bytes(source.read_bytes())

    # joblib-артефакт распиклится только при импортируемом src.*
    src_dir = out_dir / "src"
    src_dir.mkdir(exist_ok=True)
    for module in sorted((ROOT / "src").glob("*.py")):
        (src_dir / module.name).write_bytes(module.read_bytes())

    pack = ROOT / "qc_pack.zip"
    with zipfile.ZipFile(pack, "w", zipfile.ZIP_STORED) as archive:
        for item in sorted(out_dir.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(out_dir))
    print(f"{pack} ({pack.stat().st_size / 1e6:.1f} МБ), записей {len(records)}")


def pd_read_corrections() -> dict[int, int]:
    import pandas as pd

    path = ROOT / "label_corrections.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return dict(zip(frame["id"].astype(int), frame["corrected_label"].astype(int)))


if __name__ == "__main__":
    main()
