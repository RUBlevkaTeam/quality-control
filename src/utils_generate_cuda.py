"""Grounded Qwen3.5 comment rewriting with deterministic fallbacks."""

from __future__ import annotations

import gc
import re
from typing import Sequence

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"</?(?:комментарий|вердикт|rationale)>", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def _clean_generated(raw: object) -> str:
    text = _THINK_RE.sub(" ", str(raw or ""))
    text = _TAG_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip().strip('"«»')
    return text


def _prompt(
    tokenizer,
    *,
    row,
    prediction: int,
    variants: Sequence[str],
) -> str:
    verdict = "не бан" if int(prediction) == 1 else "бан"
    product_text = str(row.get("text", ""))[:2400]
    messages = [
        {
            "role": "system",
            "content": (
                "Ты выбираешь более ясную формулировку готового объяснения. "
                "Вердикт и факты менять нельзя. Ответь строго одной ASCII-буквой A или B."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Категория: {row.get('category', '')}\n"
                f"Фиксированный вердикт: {verdict}\n"
                f"Карточка: {product_text}\n"
                f"A: {variants[0]}\n"
                f"B: {variants[1]}\n"
                "Выбери более ясный вариант. Только A или B."
            ),
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def generate_grounded_comments_cuda(
    model_path: str,
    dataframe: pd.DataFrame,
    predictions: Sequence[int],
    comment_variants: Sequence[Sequence[str]],
    *,
    batch_size: int = 64,
    max_new_tokens: int = 4,
) -> list[str]:
    if not (len(dataframe) == len(predictions) == len(comment_variants)):
        raise ValueError("comment generation inputs must align")
    if len(dataframe) == 0:
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generated: list[str] = []
    for start in range(0, len(dataframe), batch_size):
        end = min(start + batch_size, len(dataframe))
        prompts = [
            _prompt(
                tokenizer,
                row=dataframe.iloc[position],
                prediction=int(predictions[position]),
                variants=comment_variants[position],
            )
            for position in range(start, end)
        ]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_length = encoded["input_ids"].shape[1]
        for sequence in output:
            decoded = tokenizer.decode(sequence[input_length:], skip_special_tokens=True)
            generated.append(_clean_generated(decoded))

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated
