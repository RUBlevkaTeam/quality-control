import gc
import re
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.constants import MAX_PROMPT_TOKENS


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
        max_length=MAX_PROMPT_TOKENS,
        return_tensors="pt",
    ).to(model.device)

    # 0 - валидный id, поэтому "or" тут не годится: он провалится дальше по цепочке
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

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
    "Ты модератор маркетплейса. Пишешь короткие объяснения вердиктов проверки "
    "товаров. Отвечай одним предложением на русском, 120-250 символов, без "
    "переносов строк, без тегов и рассуждений - только готовое объяснение."
)

# Вердикт зафиксирован классификатором до генерации - LLM объясняет, а не решает.
_USER_PROMPT_TEMPLATE = (
    "Категория проверки: {category}.\n"
    "Правило: {rule}\n"
    "Карточка товара: {text}\n\n"
    "Вердикт проверки: {verdict}. {evidence}\n"
    "Напиши одно предложение, объясняющее этот вердикт для проверяющего "
    "сотрудника со ссылкой на данные карточки. Вердикт менять нельзя."
)

_RULES = {
    "БАД": (
        "товар относится к биологически активным добавкам, если карточка "
        "содержит маркировку БАД/биологически активной добавки; спортивное "
        "питание и товары без маркировки к БАД не относятся"
    ),
    "Легковоспламеняющиеся": (
        "товар относится к легковоспламеняющимся, если он сам является "
        "горючим веществом, пиротехникой или продаётся с топливом в комплекте; "
        "пустые устройства без топлива к категории не относятся"
    ),
}


# короткая сводка найденных правилами оснований - чтобы LLM не выдумывала факты
def _evidence_summary(name: object, description: object, prediction: int) -> str:
    try:
        from src.rule_features import evidence_flags

        flags = evidence_flags(name, description)
    except Exception:
        return ""
    facts = []
    if flags.get("bad_marker"):
        facts.append("в карточке есть маркировка БАД")
    if flags.get("not_bad"):
        facts.append("есть прямое указание, что товар не является БАД")
    if flags.get("sports"):
        facts.append("товар выглядит как спортивное питание")
    if flags.get("standalone"):
        facts.append("товар - самостоятельное горючее или пиротехника")
    if flags.get("fuel") and flags.get("included"):
        facts.append("топливо или газ входит в комплект")
    if flags.get("empirical_absence_veto"):
        facts.append("топливо или газ в комплект не входит")
    if flags.get("component_or_accessory"):
        facts.append("это аксессуар или негорючий компонент")
    if not facts:
        return ""
    return "Основания: " + "; ".join(facts[:3]) + "."


# Helper to build chat template prompt for a single sample
def _build_prompt(row, tokenizer) -> str:
    prediction = int(row.get("pred", 0) or 0)
    category = str(row.get("category", ""))
    user_text = _USER_PROMPT_TEMPLATE.format(
        category=category,
        rule=_RULES.get(category, "правила категории площадки"),
        text=str(row.get("text", ""))[:2400],
        verdict="не бан" if prediction == 1 else "бан",
        evidence=_evidence_summary(row.get("name"), row.get("description"), prediction),
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
    if len(df) == 0:
        return []

    # проверка идёт офлайн: неверный путь должен падать сразу, а не уходить в сеть
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
    )

    # КРИТИЧНО для decoder-only: при padding справа короткие промпты в батче
    # продолжаются от pad-токенов, и срез out[input_length:] вырезает мусор.
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
            prompts_list.append(_build_prompt(row, tokenizer))

        # Generate in one batch call
        batch_comments = _generate_batch(
            model, tokenizer, prompts_list,
            max_new_tokens, do_sample
        )
        comments.extend(batch_comments)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return comments
