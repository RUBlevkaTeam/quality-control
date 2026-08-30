#!/usr/bin/env python3
"""Готовит компактный набор картинок для обучения LoRA на Kaggle.

Полный images/ весит 9 ГБ, но обучению столько не нужно: pick_images в
scripts/train_lora_qc.py берёт из первых шести файлов товара ДВА самых
крупных, а процессор потом ужимает их до max_pixels (512*28*28). Всё
остальное разрешение выбрасывается впустую - и при загрузке на Kaggle
стоит часов.

Поэтому оставляем те же два файла на товар и заранее приводим их к целевой
площади. Имена и структура каталогов сохраняются, так что train_lora_qc.py
и src/lora_classifier.py работают без единой правки: недостающие файлы они
и так пропускают по OSError.

Качество JPEG держим высоким (90): маркировка «биологически активная
добавка» напечатана на упаковке мелким шрифтом, и артефакты сжатия бьют
ровно по тому сигналу, ради которого визуальный канал и нужен.

Запуск:
    python scripts/pack_images_for_kaggle.py            # -> images_small/
    python scripts/pack_images_for_kaggle.py --zip      # + images_small.zip
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# те же лимиты, что у процессора при обучении
MAX_PIXELS = 512 * 28 * 28
MIN_PIXELS = 256 * 28 * 28
PROBE_LIMIT = 6      # pick_images смотрит первые шесть файлов
KEEP = 2             # ...и оставляет два крупнейших
QUALITY = 90
_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _product_images(directory: Path) -> list[Path]:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in _EXTENSIONS)
    return files[:PROBE_LIMIT]


def _convert(job: tuple[Path, Path]) -> tuple[int, int]:
    """Возвращает (записано файлов, сэкономлено байт)."""
    source_dir, target_dir = job
    try:
        candidates = []
        for path in _product_images(source_dir):
            try:
                with Image.open(path) as probe:
                    candidates.append((probe.width * probe.height, path))
            except (OSError, ValueError):
                continue
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        chosen = candidates[:KEEP]
        if not chosen:
            return 0, 0

        target_dir.mkdir(parents=True, exist_ok=True)
        written = saved = 0
        for area, path in chosen:
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if area > MAX_PIXELS:
                        scale = math.sqrt(MAX_PIXELS / area)
                        size = (max(1, int(image.width * scale)),
                                max(1, int(image.height * scale)))
                        image = image.resize(size, Image.LANCZOS)
                    # имя сохраняем: пути в train.jsonl должны сойтись
                    out = target_dir / (path.stem + ".jpg")
                    image.save(out, "JPEG", quality=QUALITY, optimize=True)
                written += 1
                saved += max(0, path.stat().st_size - out.stat().st_size)
            except (OSError, ValueError):
                continue
        return written, saved
    except OSError:
        return 0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=str(ROOT / "images"))
    # Внутри создаём подкаталог images/: пути в train.jsonl записаны как
    # "images/<id>/<file>.jpg", поэтому обучению надо передавать РОДИТЕЛЯ
    # (--images kaggle_images), а не саму папку с товарами.
    parser.add_argument("--out", default=str(ROOT / "kaggle_images"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--zip", action="store_true", help="собрать архив рядом")
    args = parser.parse_args()

    source_root = Path(args.images)
    target_root = Path(args.out)
    if not source_root.is_dir():
        raise SystemExit(f"нет каталога с картинками: {source_root}")
    products = sorted((p for p in source_root.iterdir() if p.is_dir()),
                      key=lambda p: p.name)
    if not products:
        raise SystemExit(f"в {source_root} нет папок товаров")
    print(f"товаров: {len(products)}", flush=True)

    images_root = target_root / "images"
    jobs = [(p, images_root / p.name) for p in products]
    written = saved = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for index, (count, bytes_saved) in enumerate(pool.map(_convert, jobs, chunksize=32), 1):
            written += count
            saved += bytes_saved
            if index % 2000 == 0:
                print(f"  {index}/{len(jobs)} товаров, {written} файлов", flush=True)

    # Тихо пропущенные картинки - самая дорогая ошибка здесь: обучение
    # молча станет текстовым. Поэтому проверяем сходимость путей сразу.
    pack = ROOT / "qc_pack" / "train.jsonl"
    if pack.is_file():
        import json as _json

        checked = hit = 0
        with pack.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if line_number >= 300:
                    break
                for rel in _json.loads(line).get("image_paths", [])[:PROBE_LIMIT]:
                    checked += 1
                    hit += (target_root / rel).is_file()
        share = hit / max(checked, 1)
        print(f"проверка путей: {hit}/{checked} файлов на месте ({share:.0%})")
        if share < 0.2:
            raise SystemExit(
                f"пути не сходятся - обучение пойдёт без картинок; "
                f"передавайте --images {target_root}"
            )

    total = sum(f.stat().st_size for f in target_root.rglob("*.jpg"))
    print(f"\nзаписано {written} картинок в {target_root}")
    print(f"объём: {total/1e9:.2f} ГБ (было 9 ГБ, экономия {saved/1e9:.2f} ГБ)")

    if args.zip:
        archive = shutil.make_archive(str(target_root), "zip", root_dir=target_root)
        print(f"архив: {archive} ({Path(archive).stat().st_size/1e9:.2f} ГБ)")


if __name__ == "__main__":
    main()
