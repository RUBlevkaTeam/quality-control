import re
import sys
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.constants import MAX_PROMPT_TOKENS
from src.device import (
    out_of_memory_errors,
    describe,
    free_memory,
    model_placement,
    select_device,
    select_dtype,
)


def _log(message: str) -> None:
    print(f"[generate] {message}", file=sys.stderr, flush=True)


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
    "переносов строк, без тегов и рассуждений - только готовое объяснение. "
    "Всегда называй конкретную формулировку из карточки, на которой основано "
    "решение. Общие фразы вроде «найдены признаки нарушений» недопустимы. "
    "Слова «бан» и «не бан» в объяснении не используй - пиши «относится» или "
    "«не относится» к категории. Объяснение обязано ПОДДЕРЖИВАТЬ вердикт: "
    "если вердикт «товар не относится», нельзя приводить цитату как "
    "доказательство обратного.\n\n"
    "Официальные правила категорий:\n\n"
    "Правила отнесения товара к биологически активным добавкам.\n"
    "Товар является биологически активной добавкой, если: в описании или на "
    "изображении содержится прямое указание, что товар является биологически "
    "активной добавкой (БАД, dietary supplement).\n"
    "Товар не является биологически активной добавкой, если: товар является "
    "спортивным питанием, таким как: аминокислоты, BCAA, L-карнитин, протеин "
    "или иным товаром с прямым указанием на принадлежность к спортивному "
    "питанию; в описании явно указано что товар не является биологически "
    "активной добавкой; товар не содержит маркировок биологически активной "
    "добавки (БАД, dietary supplement).\n\n"
    "Правила отнесения товара к легковоспламеняющимся.\n"
    "Товар является легковоспламеняющимся, если: товар является "
    "самостоятельным источником воспламенения - изделия, основное назначение "
    "которых — создание или поддержание открытого огня, например, спички, "
    "зажигалки; товар содержит горючее вещество, легковоспламеняющиеся "
    "вещества или горючие газы; в комплект товара входит легковоспламеняющийся "
    "товар.\n"
    "Товар не является легковоспламеняющимся, если: товар не содержит "
    "источника воспламенения или горючего вещества - устройство, "
    "предназначенное для использования с огнем или горючими веществами, само "
    "по себе не является легковоспламеняющимся, например мангалы, грили, "
    "газовые плиты; легковоспламеняющимся является содержимое, а не сама "
    "конструкция - товар не считается легковоспламеняющимся при отсутствии "
    "содержимого; источник воспламенения встроен в изделие; потенциально "
    "легковоспламеняющийся материал используется как компонент изделия, "
    "например, активированный уголь в фильтрах, уголь для рисования; "
    "легковоспламеняющийся предмет не входит в комплект."
)

# Few-shot: структура «цитата из карточки -> связка с правилом -> вывод».
# Пары user/assistant в формате боевого промпта; ответы не содержат слов
# «бан»/«не бан» (вердикт идёт отдельным тегом) и всегда согласованы с ним.
_FEWSHOT = (
    (
        "Категория проверки: БАД.\n"
        "Карточка товара: Название: Коллаген морской с витамином C, 60 капсул "
        "Описание: Биологически активная добавка к пище. БАД. Не является "
        "лекарственным средством.\n\n"
        "Вердикт проверки: не бан - это значит, что товар ДЕЙСТВИТЕЛЬНО относится "
        "к категории «БАД». В карточке указано: «БАД. Не является лекарственным "
        "средством» (маркировка БАД).\n"
        "Напиши одно предложение для проверяющего сотрудника: объясни вердикт и "
        "ОБЯЗАТЕЛЬНО назови конкретные слова из карточки, на которых он основан. "
        "Не пиши общих формулировок - укажи, что именно написано в этом товаре. "
        "Вердикт менять нельзя.",
        "Товар отнесён к категории верно: в описании есть обязательная маркировка "
        "«Биологически активная добавка к пище. БАД. Не является лекарственным "
        "средством», что по правилу площадки прямо определяет БАД.",
    ),
    (
        "Категория проверки: БАД.\n"
        "Карточка товара: Название: Креатин моногидрат 500 г, спортивное питание "
        "Описание: Чистый креатин для набора силы и массы, для спортсменов.\n\n"
        "Вердикт проверки: бан - это значит, что товар НЕ относится к категории "
        "«БАД» и попал в неё ошибочно. В карточке указано: «спортивное питание» "
        "(спортивное питание).\n"
        "Напиши одно предложение для проверяющего сотрудника: объясни вердикт и "
        "ОБЯЗАТЕЛЬНО назови конкретные слова из карточки, на которых он основан. "
        "Не пиши общих формулировок - укажи, что именно написано в этом товаре. "
        "Вердикт менять нельзя.",
        "Товар не относится к БАД: в названии прямо указано «спортивное питание», "
        "а по правилу площадки спортивное питание (креатин, протеин, аминокислоты) "
        "к биологически активным добавкам не относится, маркировки БАД в карточке нет.",
    ),
    (
        "Категория проверки: Легковоспламеняющиеся.\n"
        "Карточка товара: Название: Горелка туристическая с газовым баллоном 220 г "
        "Описание: Портативная горелка, газовый баллон в комплекте.\n\n"
        "Вердикт проверки: не бан - это значит, что товар ДЕЙСТВИТЕЛЬНО относится "
        "к категории «Легковоспламеняющиеся». В карточке указано: «газовый баллон "
        "в комплекте» (топливо в комплекте).\n"
        "Напиши одно предложение для проверяющего сотрудника: объясни вердикт и "
        "ОБЯЗАТЕЛЬНО назови конкретные слова из карточки, на которых он основан. "
        "Не пиши общих формулировок - укажи, что именно написано в этом товаре. "
        "Вердикт менять нельзя.",
        "Товар отнесён к категории верно: в описании указано «газовый баллон в "
        "комплекте», то есть вместе с устройством продаётся горючий газ, а товары "
        "с топливом в комплекте относятся к легковоспламеняющимся.",
    ),
    (
        "Категория проверки: Легковоспламеняющиеся.\n"
        "Карточка товара: Название: Опахало для мангала, веер для раздува углей "
        "Описание: Деревянное опахало, ускоряет розжиг углей.\n\n"
        "Вердикт проверки: бан - это значит, что товар НЕ относится к категории "
        "«Легковоспламеняющиеся» и попал в неё ошибочно.\n"
        "Напиши одно предложение для проверяющего сотрудника: объясни вердикт и "
        "ОБЯЗАТЕЛЬНО назови конкретные слова из карточки, на которых он основан. "
        "Не пиши общих формулировок - укажи, что именно написано в этом товаре. "
        "Вердикт менять нельзя.",
        "Товар не относится к категории: это «опахало для мангала, веер для "
        "раздува углей» - аксессуар для работы с огнём, сам он не горючее вещество, "
        "и никакого топлива в комплекте карточка не упоминает.",
    ),
    (
        "Категория проверки: Легковоспламеняющиеся.\n"
        "Карточка товара: Название: Плита газовая портативная для кемпинга "
        "Описание: Компактная плита под цанговый баллон. Внимание: баллон в "
        "комплект не входит!\n\n"
        "Вердикт проверки: бан - это значит, что товар НЕ относится к категории "
        "«Легковоспламеняющиеся» и попал в неё ошибочно. В карточке указано: "
        "«баллон в комплект не входит» (топливо не входит в комплект).\n"
        "Напиши одно предложение для проверяющего сотрудника: объясни вердикт и "
        "ОБЯЗАТЕЛЬНО назови конкретные слова из карточки, на которых он основан. "
        "Не пиши общих формулировок - укажи, что именно написано в этом товаре. "
        "Вердикт менять нельзя.",
        "Товар не относится к категории: в описании прямо сказано «баллон в "
        "комплект не входит», то есть продаётся только негорючее устройство без "
        "топлива, а такие товары по правилу к легковоспламеняющимся не относятся.",
    ),
)

# Вердикт зафиксирован классификатором до генерации - LLM объясняет, а не решает.
# Что означают вердикты. Без этого модель читает «бан» как «заблокировать
# опасный товар» и пишет самопротиворечиво: объясняет, почему товар горюч,
# и тут же выносит «не бан». По данным смысл обратный: проверяется, верно ли
# товар отнесён к категории (в «Легковоспламеняющихся» 96% товаров к ней
# не относятся - это мангалы и аксессуары без топлива).
_VERDICT_MEANING = {
    "не бан": "товар ДЕЙСТВИТЕЛЬНО относится к категории «{category}»",
    "бан": "товар НЕ относится к категории «{category}» и попал в неё ошибочно",
}

# правило категории не повторяется в каждом сообщении: полный официальный
# текст один раз лежит в системном промпте
_USER_PROMPT_TEMPLATE = (
    "Категория проверки: {category}.\n"
    "Карточка товара: {text}\n\n"
    "Вердикт проверки: {verdict} - это значит, что {meaning}. {evidence}\n"
    "Напиши одно предложение для проверяющего сотрудника: объясни вердикт и "
    "ОБЯЗАТЕЛЬНО назови конкретные слова из карточки, на которых он основан. "
    "Не пиши общих формулировок - укажи, что именно написано в этом товаре. "
    "Вердикт менять нельзя."
)


# короткая сводка найденных правилами оснований - чтобы LLM не выдумывала факты
def _evidence_summary(
    name: object, description: object, prediction: int, category: object = ""
) -> str:
    """Подсказка для LLM: сначала конкретные цитаты из карточки, и лишь при
    их отсутствии - обобщённые основания.

    Ручная проверка организаторов оценивает, названо ли конкретное место
    товара, поэтому цитата ценнее пересказа правила."""
    try:
        from src.rule_features import evidence_flags, evidence_quotes
    except Exception:
        return ""
    try:
        quotes = evidence_quotes(name, description, category)
        if quotes:
            parts = "; ".join(f"«{quote}» ({label})" for label, quote in quotes)
            return f"В карточке указано: {parts}."
    except Exception:
        pass
    try:
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
    verdict = "не бан" if prediction == 1 else "бан"
    user_text = _USER_PROMPT_TEMPLATE.format(
        category=category,
        text=str(row.get("text", ""))[:2400],
        verdict=verdict,
        meaning=_VERDICT_MEANING[verdict].format(category=category),
        evidence=_evidence_summary(
            row.get("name"), row.get("description"), prediction, category
        ),
    )

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for shot_user, shot_answer in _FEWSHOT:
        messages.append({"role": "user", "content": shot_user})
        messages.append({"role": "assistant", "content": shot_answer})
    messages.append({"role": "user", "content": user_text})

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

    # cuda -> mps -> cpu: в контейнере всегда cuda, mps нужен для локальной
    # отладки промптов на макбуке
    device = select_device()
    print(f"[generate] устройство: {describe(device)}", file=sys.stderr, flush=True)

    # проверка идёт офлайн: неверный путь должен падать сразу, а не уходить в сеть
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype=select_dtype(device),
        local_files_only=True,
        trust_remote_code=True,
        **model_placement(device),
    )
    # без device_map (не-cuda) модель остаётся на CPU - переносим руками
    if device != "cuda":
        model = model.to(device)

    # КРИТИЧНО для decoder-only: при padding справа короткие промпты в батче
    # продолжаются от pad-токенов, и срез out[input_length:] вырезает мусор.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    n = len(df)

    # Промпты строим заранее: батч паддится до самого длинного в нём, поэтому
    # соседство коротких карточек с длинными - это чистый перерасход. Замер на
    # T4: порядок «как в df» 47 мин против 40 мин по возрастанию длины (-17%).
    prompts_all = [_build_prompt(row, tokenizer) for _, row in df.iterrows()]
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts_all]
    order = sorted(range(n), key=lambda i: lengths[i])

    comments_by_index: List[str] = [""] * n
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]

        # На OOM режем батч пополам и повторяем: у Qwen3.5 linear attention
        # поднимает тензоры в fp32, и фактический потолок ниже расчётного.
        step = len(idx)
        while idx:
            chunk, rest = idx[:step], idx[step:]
            try:
                batch_comments = _generate_batch(
                    model, tokenizer, [prompts_all[i] for i in chunk],
                    max_new_tokens, do_sample
                )
            except out_of_memory_errors():
                free_memory(device)
                if step == 1:
                    # один патологический промпт не должен стирать уже
                    # сгенерированный тираж: пропускаем товар (пустая строка
                    # уйдёт в детерминированный фолбэк), едем дальше
                    _log(f"OOM на одиночном товаре {chunk[0]}, пропускаю его")
                    idx = rest
                    continue
                step = max(1, step // 2)
                _log(f"OOM при генерации, уменьшаю батч до {step}")
                continue
            for position, source in zip(chunk, batch_comments):
                comments_by_index[position] = source
            idx = rest

    comments: List[str] = comments_by_index

    del model, tokenizer
    free_memory(device)

    return comments
