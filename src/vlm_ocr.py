"""VLM-транскрипция изображений на инференсе (маркировка живёт на упаковке).

Классификатор потребляет колонку ``ocr`` так же, как при обучении
(scripts/build_ocr_texts.py готовит train-сторону тем же промптом и тем же
лимитом пикселей). Здесь три предохранителя:

1. гейт по неуверенности: транскрибируем только товары категории
   «Легковоспламеняющиеся» и спорную зону БАД около порога;
2. бюджет времени: как только он исчерпан, остаток получает пустой ocr;
3. любая ошибка деградирует до пустых строк, никогда не роняет прогон.
"""

from __future__ import annotations

import sys
import time
from typing import List, Sequence

from src.constants import DEFAULT_OCR_MAX_PIXELS

_FIRE_CATEGORY = "Легковоспламеняющиеся"
_GATE_DELTA = 0.15

_PROMPT = (
    "Перечисли текст, который написан на товаре или упаковке на фото: название, "
    "назначение, предупреждения. Только реально видимый текст, без комментариев, "
    "кратко. Если текста нет - напиши: нет текста."
)


def _log(message: str) -> None:
    print(f"[ocr] {message}", file=sys.stderr, flush=True)


def _clean(text: str) -> str:
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return " ".join(text.split())[:1200]


def _pick_images(paths, limit: int) -> List[str]:
    from PIL import Image

    def area(path) -> int:
        try:
            with Image.open(path) as im:
                return im.width * im.height
        except OSError:
            return 0

    ranked = sorted(paths, key=area, reverse=True)
    return [p for p in ranked if area(p) > 0][:limit]


def candidate_mask(df, probabilities: Sequence[float], thresholds: dict) -> "object":
    """Кому нужна транскрипция: fire всем, остальным - спорная зона у порога."""
    import numpy as np

    probs = np.asarray(probabilities, dtype=np.float64)
    mask = np.zeros(len(df), dtype=bool)
    categories = df["category"].astype(str).tolist() if "category" in df.columns else [""] * len(df)
    for i, category in enumerate(categories):
        threshold = float(thresholds.get(category, 0.5))
        if category == _FIRE_CATEGORY or abs(probs[i] - threshold) <= _GATE_DELTA:
            mask[i] = True
    has_images = df["n_images"] > 0 if "n_images" in df.columns else False
    return mask & np.asarray(has_images, dtype=bool)


def transcribe_products(
    df,
    positions: Sequence[int],
    *,
    model_path: str,
    budget_seconds: float = 600.0,
    images_per_product: int = 2,
    batch_size: int = 8,
    max_new_tokens: int = 80,
) -> List[str]:
    """Возвращает список строк ocr длиной len(df); без картинок - пустая строка."""

    n = len(df)
    output = [""] * n
    positions = [p for p in positions if df["image_paths"].iloc[p]]
    if not positions or budget_seconds <= 0:
        return output

    started = time.time()
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, max_pixels=DEFAULT_OCR_MAX_PIXELS
        )
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, local_files_only=True
        ).to(device)
        model.eval()
    except Exception as exc:
        _log(f"модель недоступна ({type(exc).__name__}: {exc}), ocr пропущен")
        return output

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}],
        }
    ]
    try:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def expand(text: str, count: int) -> str:
        return text * max(count, 1)

    done = 0
    for start in range(0, len(positions), batch_size):
        remaining_budget = budget_seconds - (time.time() - started)
        if remaining_budget <= 0:
            _log(f"бюджет исчерпан после {done} товаров")
            break
        chunk = positions[start : start + batch_size]
        batch_images = []
        batch_positions = []
        for position in chunk:
            paths = _pick_images(list(df["image_paths"].iloc[position]), images_per_product)
            opened = []
            for path in paths:
                try:
                    opened.append(_open_rgb(path))
                except Exception:
                    continue
            if opened:
                batch_images.append(opened)
                batch_positions.append(position)
        if not batch_images:
            continue
        try:
            inputs = processor(
                text=[expand(prompt, len(images)) for images in batch_images],
                images=batch_images,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            texts = processor.tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        except Exception as exc:
            _log(f"батч пропущен ({type(exc).__name__}: {exc})")
            texts = [""] * len(batch_images)
        for position, images, text in zip(batch_positions, batch_images, texts):
            output[position] = _clean(text)
            for image in images:
                image.close()
        done += len(batch_positions)

    _log(
        f"транскрибировано {done} товаров за {time.time() - started:.0f}s "
        f"(непустых {sum(1 for t in output if t)})"
    )
    return output


def _open_rgb(path):
    from PIL import Image

    with Image.open(path) as source:
        return source.convert("RGB")
