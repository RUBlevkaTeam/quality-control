#!/usr/bin/env python3
"""Эквивалентность двух способов применить адаптер: peft против запасного
пути из src/lora_classifier (LoRA-хуки на лету).

Запасной путь поедет в контейнер (peft там может не быть), и до отправки
обязан дать ТЕ ЖЕ вероятности, что peft. Проверяем именно боевую функцию
_attach_adapter_hooks, а не её копию. Сравниваем логиты токенов «да»/«нет»
на нескольких промптах.

История: прежний запасной путь ВЛИВАЛ веса (W += (alpha/r)·B@A) - дельта
дважды округлялась до bf16, расхождение вероятностей доходило до 0.03.
Хуки повторяют арифметику peft и обязаны сходиться до <0.005.

Запуск на сервере (peft там есть):
    python3 verify_lora_merge.py --model qwen --adapter qc_lora_final/adapter_all
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE if (_HERE / "src").is_dir() else _HERE.parent))

from src.lora_classifier import _attach_adapter_hooks

PROMPTS = [
    "Категория проверки: Легковоспламеняющиеся.\nНазвание: Горелка газовая с баллоном\nОтносится ли товар к категории? Ответь одним словом: да или нет.",
    "Категория проверки: БАД.\nНазвание: Витамин C 500мг капсулы, биологически активная добавка\nОтносится ли товар к категории? Ответь одним словом: да или нет.",
    "Категория проверки: Легковоспламеняющиеся.\nНазвание: Опахало для мангала\nОтносится ли товар к категории? Ответь одним словом: да или нет.",
]


def yes_no_logits(model, processor, device):
    tok = processor.tokenizer
    yes = tok("да", add_special_tokens=False)["input_ids"][0]
    no = tok("нет", add_special_tokens=False)["input_ids"][0]
    out = []
    for text in PROMPTS:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        try:
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[rendered], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].float()
        out.append((float(logits[yes]), float(logits[no])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--adapter", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)

    print("=== путь 1: peft ===")
    from peft import PeftModel

    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, local_files_only=True, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).to(device).eval()
    ref = yes_no_logits(model, processor, device)
    del model, base
    gc.collect(); torch.cuda.empty_cache()

    print("=== путь 2: LoRA-хуки (боевой запасной путь) ===")
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, local_files_only=True, attn_implementation="sdpa")
    attached = _attach_adapter_hooks(base, args.adapter, torch)
    print(f"хуков навешено: {attached}")
    model = base.to(device).eval()
    ours = yes_no_logits(model, processor, device)

    worst = 0.0
    for i, ((y1, n1), (y2, n2)) in enumerate(zip(ref, ours)):
        p1 = 1 / (1 + torch.exp(torch.tensor(n1 - y1))).item()
        p2 = 1 / (1 + torch.exp(torch.tensor(n2 - y2))).item()
        diff = abs(p1 - p2)
        worst = max(worst, diff)
        print(f"  промпт {i}: peft p={p1:.4f} | ручное p={p2:.4f} | разница {diff:.5f}")
    print(f"\nмакс расхождение вероятностей: {worst:.5f}")
    if worst < 0.005:
        print("ЭКВИВАЛЕНТНО: запасной путь можно отправлять")
    else:
        print("РАСХОДИТСЯ: в контейнер с таким нельзя, разбираться")


if __name__ == "__main__":
    main()
