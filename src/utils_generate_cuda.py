import gc
import os
from typing import List

import numpy as np
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Helper to tokenize a batch of prompts and generate comments in one pass
def _generate_batch(
    model,
    tokenizer,
    prompts_list: List[str],
    max_new_tokens: int,
    do_sample: bool,
) -> List[str]:
    
    encoded = tokenizer(
        prompts_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    pad_token_id = model.config.pad_token_id or tokenizer.pad_token_id or 151643

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            use_cache=True,
        )

    input_length = encoded['input_ids'].shape[1]
    comments = []
    for out in outputs:
        generated_tokens = out[input_length:]
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        comments.append(_clean_output(raw_text))
    return comments


_SYSTEM_PROMPT = (
    "You are a helpful product analyst. Be concise and strictly follow formatting instructions. "
    "Do NOT output any thinking process, reasoning, or thinking tags. Output only the final answer."
)

_USER_PROMPT_TEMPLATE = (
    "Перед нами {pred_label} продукт в онлайн магазине.\n"
    "Описание продукта: {text}\n\n"
    "Напиши краткое объяснение почему этот продукт {pred_label_lower} за 30 слов или короче. "
    "Используй русский язык."
)


# Helper to build chat template prompt for a single sample
def _build_prompt(text: str, prediction: int, tokenizer) -> str:
    
    pred_label = "хороший" if prediction else "плохой"
    pred_label_lower = pred_label.lower()

    user_text = _USER_PROMPT_TEMPLATE.format(
        pred_label=pred_label,
        pred_label_lower=pred_label_lower,
        text=text,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


# Strip reasoning tags and extract <rationale> content if present
def _clean_output(raw: str) -> str:
    
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "<rationale>" in text:
        match = re.search(r"<rationale>(.*?)</rationale>", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text


# Load LLM model and generate comments with batched inference
def generate_comments_cuda(
    llm_model_path: str,
    df: pd.DataFrame,
    batch_size: int = 64,
    max_new_tokens: int = 120,
    do_sample: bool = False,
) -> List[str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    _is_local = os.path.exists(llm_model_path) or (
        os.path.isabs(llm_model_path) and not llm_model_path.startswith(("http://", "https://", "file://"))
    )
    load_kwargs = {
        "local_files_only": _is_local,
        "trust_remote_code": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        dtype=dtype,
        **load_kwargs,
    ).to(device)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n = len(df)
    comments: List[str] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_df = df.iloc[start:end]

        # Collect prompts
        prompts_list = []
        for _, row in batch_df.iterrows():
            text = row.get('text', '') or ''
            prediction = int(row.get('pred', 0))
            prompts_list.append(_build_prompt(text, prediction, tokenizer))

        # Generate in one batch call
        batch_comments = _generate_batch(
            model, tokenizer, prompts_list,
            max_new_tokens, do_sample
        )
        comments.extend(batch_comments)

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return comments
