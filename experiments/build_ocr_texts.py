#!/usr/bin/env python3
"""Транскрипция изображений товаров через локальный VLM Qwen3.5-4B.

Модель мультимодальная (Qwen3_5ForConditionalGeneration), маркировка «БАД»,
состав и предупреждения живут на упаковке - OCR-текст даёт классификатору
то, что не доехало в описание. Скрипт офлайн: готовит ocr_texts.jsonl для
train; в контейнере тот же промпт исполняется в run.py для теста.

Запуск (пилот):
  venv312/bin/python scripts/build_ocr_texts.py --category Легковоспламеняющиеся --limit 20

Полный прогон возобновляемый: каждая запись дописывается в jsonl, при
перезапуске уже готовые id пропускаются.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from src.utils_data_prep import prepare_dataframe

DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen3.5-4B"

# Строгая инструкция без места для рассуждений: модель думающая, поэтому
# просим только перечисление надписей и выключаем thinking на уровне шаблона.
_PROMPT = (
    "Перечисли текст, который написан на товаре или упаковке на фото: название, "
    "назначение, предупреждения. Только реально видимый текст, без комментариев, "
    "кратко. Если текста нет - напиши: нет текста."
)
_MAX_IMAGES_PER_PRODUCT = 2


def _pick_images(paths: list[str], limit: int) -> list[str]:
    """Самые крупные изображения: на них читаемее мелкая маркировка."""

    def area(path: str) -> int:
        try:
            with Image.open(path) as im:
                return im.width * im.height
        except OSError:
            return 0

    ranked = sorted(paths, key=area, reverse=True)
    return [p for p in ranked if area(p) > 0][:limit]


def _build_prompt(processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}],
        }
    ]
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _clean(text: str) -> str:
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return " ".join(text.split())[:1200]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", default=[], dest="categories")
    parser.add_argument("--limit", type=int, default=0, help="максимум товаров (пилот)")
    parser.add_argument("--images-per-product", type=int, default=_MAX_IMAGES_PER_PRODUCT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=110)
    parser.add_argument(
        "--out", default=str(ROOT / "ocr_texts.jsonl"), help="выходной jsonl"
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--strategy",
        choices=("all", "bad_no_marker"),
        default="all",
        help="bad_no_marker: только БАД с label=1 без маркировки в тексте",
    )
    parser.add_argument(
        "--ids-file", default="",
        help="csv с колонкой id: обработать только эти товары (пилот по ошибкам OOF)",
    )
    parser.add_argument("--data", default=str(ROOT / "data.csv"))
    parser.add_argument("--images", default=str(ROOT / "images"))
    args = parser.parse_args()

    df = prepare_dataframe(args.data, args.images)
    if args.categories:
        df = df[df["category"].isin(args.categories)]
    if args.ids_file:
        wanted = set(pd.read_csv(args.ids_file)["id"])
        df = df[df["id"].isin(wanted)]
        print(f"фильтр --ids-file: {len(df)} товаров", flush=True)
    if args.strategy == "bad_no_marker":
        import re

        from src.utils_data_prep import _clean as _clean_series

        marker = re.compile(
            r"(?<![а-яa-z])бад(?![а-яa-z])|биологически\s+активн|dietary\s+supplement"
        )
        bad = df[(df["category"] == "БАД") & (df["label"] == 1)]
        marked = _clean_series(bad["name"] + " " + bad["description"]).str.contains(
            marker, case=False, regex=True
        )
        df = bad[~marked]
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out)
    done: set[int] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["id"]))
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
    pending = df[~df["id"].isin(done)]
    print(f"товаров к обработке: {len(pending)} (уже готово {len(done)})", flush=True)
    if len(pending) == 0:
        return

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    print(f"устройство: {device}", flush=True)

    # Дефолтный препроцессор режет до ~14k визуальных токенов на картинку -
    # префилл на этом съедает минуты. Ограничиваем: маркировку читает и с 400к пикселей.
    _OCR_MAX_PIXELS = 512 * 28 * 28
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_dir,
            local_files_only=True,
            max_pixels=_OCR_MAX_PIXELS,
        )
    except (TypeError, ValueError):
        processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = _OCR_MAX_PIXELS
    # decoder-only: паддинг слева, иначе генерация продолжается от pad-токенов
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()
    prompt = _build_prompt(processor)

    def _expand(text: str, count: int) -> str:
        """Число плейсхолдеров изображения должно совпадать с числом картинок."""
        return text * max(count, 1)

    started = time.time()
    processed = 0
    with out_path.open("a", encoding="utf-8") as sink:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending.iloc[start : start + args.batch_size]
            batch_images: list[list[Image.Image]] = []
            batch_ids: list[int] = []
            for _, row in chunk.iterrows():
                paths = _pick_images(list(row["image_paths"]), args.images_per_product)
                opened = []
                for p in paths:
                    try:
                        opened.append(Image.open(p).convert("RGB"))
                    except OSError:
                        continue
                if opened:
                    batch_images.append(opened)
                    batch_ids.append(int(row["id"]))

            if not batch_images:
                continue
            try:
                inputs = processor(
                    text=[_expand(prompt, len(images)) for images in batch_images],
                    images=batch_images,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                texts = processor.tokenizer.batch_decode(
                    output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            except Exception as exc:  # битая картинка или OOM не должны ронять прогон
                print(f"[warn] батч пропущен: {type(exc).__name__}: {exc}", flush=True)
                for images in batch_images:
                    for image in images:
                        image.close()
                continue

            for product_id, images, text in zip(batch_ids, batch_images, texts):
                sink.write(
                    json.dumps(
                        {"id": product_id, "ocr": _clean(text)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                for image in images:
                    image.close()
            sink.flush()
            processed += len(batch_images)
            rate = processed / max(time.time() - started, 1e-6)
            eta_min = (len(pending) - processed) / max(rate, 1e-6) / 60
            print(
                f"{processed}/{len(pending)} ({rate:.2f} тов/с, ETA {eta_min:.0f} мин)",
                flush=True,
            )

    print(f"готово за {(time.time() - started) / 60:.1f} мин -> {out_path}")


if __name__ == "__main__":
    main()
