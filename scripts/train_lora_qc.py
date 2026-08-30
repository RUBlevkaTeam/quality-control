#!/usr/bin/env python3
"""LoRA-обучение Qwen3.5-4B (vision) как классификатора бан/не бан.

Сторона Colab/Kaggle (T4 16GB / P100): ноутбук пользователя не участвует.
Данные: qc_pack.zip (метаданные) + папка images (Kaggle Dataset или Drive).

Что делает:
  1. family-disjoint сплит (StratifiedGroupKFold по families.csv) - фолд
     целиком невидим при обучении, OOF честный;
  2. промпт: категория + правило + название + описание + до 2 крупнейших
     фото; таргет - ОДИН токен да/нет (completion-only loss);
  3. LoRA r=16 на языковых q/k/v/o проекциях, vision-tower заморожен;
  4. на валидации вероятность = softmax по логитам токенов {да, нет};
  5. печатает per-category P/R/F1 и соревновательную метрику,
     сохраняет adapter + oof_predictions.csv.

Запуск:
  python train_lora_qc.py --images /content/images --fold 0 --epochs 1
Финальный адаптер после подбора: --train-all (учится на всех фолдах).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset

PEFT = True
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    PEFT = False

YES_TOKENS = ("да", " Да", "да.")
NO_TOKENS = ("нет", " Нет", "нет.")

SYSTEM_PROMPT = (
    "Ты модератор маркетплейса. По карточке товара и фотографиям решаешь, "
    "относится ли товар к указанной категории. Отвечай одним словом."
)

# Версии правил в промпте. v1 - сжатый пересказ (боевой адаптер обучен на нём),
# v2 - дословный текст из условий хакатона. Тренер пишет rules_version.txt в
# папку адаптера, инференс (lora_classifier) читает маркер и берёт ту же
# версию - рассинхрон train/inference исключён по построению.
# Литералы RULES_VERSIONS обязаны побайтово совпадать с lora_classifier.py.
RULES_V2 = {
    "БАД": (
        "Товар является биологически активной добавкой, если:\n"
        "- В описании или на изображении содержится прямое указание, что товар "
        "является биологически активной добавкой (БАД, dietary supplement).\n"
        "Товар не является биологически активной добавкой, если:\n"
        "- Товар является спортивным питанием, таким как: аминокислоты, BCAA, "
        "L-карнитин, протеин или иным товаром с прямым указанием на "
        "принадлежность к спортивному питанию;\n"
        "- В описании явно указано что товар не является биологически активной "
        "добавкой;\n"
        "- Товар не содержит маркировок биологически активной добавки "
        "(БАД, dietary supplement)."
    ),
    "Легковоспламеняющиеся": (
        "Товар является легковоспламеняющимся, если:\n"
        "- Товар является самостоятельным источником воспламенения. К таким "
        "товарам относятся изделия, основное назначение которых — создание или "
        "поддержание открытого огня. Например, спички, зажигалки;\n"
        "- Товар содержит горючее вещество, легковоспламеняющиеся вещества или "
        "горючие газы;\n"
        "- В комплект товара входит легковоспламеняющийся товар.\n"
        "Товар не является легковоспламеняющимся, если:\n"
        "- Товар не содержит источника воспламенения или горючего вещества. "
        "Устройство, предназначенное для использования с огнем или горючими "
        "веществами, само по себе не является легковоспламеняющимся. Например "
        "мангалы, грили, газовые плиты;\n"
        "- Легковоспламеняющимся является содержимое, а не сама конструкция. "
        "Товар не считается легковоспламеняющимся при отсутствии содержимого;\n"
        "- Источник воспламенения встроен в изделие;\n"
        "- Потенциально легковоспламеняющийся материал используется как "
        "компонент изделия. Если горючий материал используется только как часть "
        "другого изделия и не является самостоятельным товаром, товар не "
        "считается легковоспламеняющимся. Например, активированный уголь в "
        "фильтрах, уголь для рисования;\n"
        "- Легковоспламеняющимся предмет не входит в комплект."
    ),
}

RULES = {
    # Официальные правила из условий хакатона - дословно, в обоих
    # файлах (train и inference) текст ОБЯЗАН совпадать.
    "БАД": (
        'Товар ЯВЛЯЕТСЯ биологически активной добавкой, если в описании или на изображении содержится прямое указание, что товар - биологически активная добавка (БАД, dietary supplement). Товар НЕ является биологически активной добавкой, если: это спортивное питание (аминокислоты, BCAA, L-карнитин, протеин или иной товар с прямым указанием на спортивное питание); в описании явно указано, что товар не является БАД; товар не содержит маркировок биологически активной добавки (БАД, dietary supplement).'
    ),
    "Легковоспламеняющиеся": (
        'Товар ЯВЛЯЕТСЯ легковоспламеняющимся, если: он самостоятельный источник воспламенения, то есть его основное назначение - создание или поддержание открытого огня (спички, зажигалки); он содержит горючее вещество, легковоспламеняющиеся вещества или горючие газы; в комплект товара входит легковоспламеняющийся товар. Товар НЕ является легковоспламеняющимся, если: он не содержит источника воспламенения или горючего вещества - устройство для использования с огнём или горючим само по себе не легковоспламеняющееся (мангалы, грили, газовые плиты); легковоспламеняющимся является содержимое, а не конструкция - без содержимого товар не считается; источник воспламенения встроен в изделие; горючий материал используется лишь как компонент изделия (активированный уголь в фильтрах, уголь для рисования); легковоспламеняющийся предмет не входит в комплект.'
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default="qc_pack")
    parser.add_argument("--images", default="images", help="корень с папками <id>/")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds-total", type=int, default=5)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--single-gpu", action="store_true",
                        help="не раскладывать модель по картам (для отладки)")
    parser.add_argument("--grad-checkpointing", action="store_true", default=True,
                        help="экономия памяти: активации пересчитываются (нужно на T4)")
    parser.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                        action="store_false")
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--desc-chars", type=int, default=700)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--fire-positive-oversample", type=int, default=3)
    parser.add_argument("--out", default="qc_lora_out")
    parser.add_argument("--category", default="",
                        help="учить только одну категорию (пусто = обе)")
    parser.add_argument("--rules", default="v1", choices=["v1", "v2"],
                        help="v1 - сжатый пересказ правил, v2 - дословный текст условия")
    parser.add_argument("--lora-targets", default="attn", choices=["attn", "all"],
                        help="attn = q/k/v/o (боевой); all = + gate/up/down MLP")
    parser.add_argument("--seed", type=int, default=2026,
                        help="сид инициализации LoRA и шаффла (для сид-супа)")
    return parser.parse_args()


def load_records(pack_dir: Path) -> list[dict]:
    # splitlines() режет ещё и по U+2028/U+2029/\x85 - в описаниях товаров
    # такие символы встречаются (27 штук на 12971 запись), а JSON считает их
    # обычным текстом. Итог - разорванная строка и JSONDecodeError.
    raw = (pack_dir / "train.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.split("\n") if line.strip()]
    fam_frame = pd_read_csv_safe(pack_dir / "families.csv")
    fam_by_id = dict(zip(fam_frame["id"].astype(int), fam_frame["family"]))
    for record in records:
        record["family"] = fam_by_id.get(int(record["id"]), -1)
    return records


def pd_read_csv_safe(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def dedup(records: list[dict]) -> list[dict]:
    seen = set()
    kept = []
    for record in records:
        key = (
            "".join(ch for ch in str(record["name"]).lower() if ch.isalnum())
            + "||"
            + "".join(ch for ch in str(record["description"]).lower() if ch.isalnum())
            ,
            int(record["label"]),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    return kept


def pick_images(paths: list[str], root: Path, limit: int) -> list[Image.Image]:
    candidates = []
    for rel in paths[:6]:
        path = root / rel
        try:
            with Image.open(path) as probe:
                area = probe.width * probe.height
                image = probe.convert("RGB").copy()
        except OSError:
            continue
        candidates.append((area, image))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    chosen = [image for _, image in candidates[:limit]]
    for _, image in candidates[limit:]:
        image.close()
    return chosen


class QcDataset(Dataset):
    def __init__(self, records, processor, images_root: Path, args, training: bool, include_answer: bool = True):
        self.records = self._balance(records, args) if training else records
        self.processor = processor
        self.root = images_root
        self.args = args
        self.training = training
        self.include_answer = include_answer

    @staticmethod
    def _balance(records, args):
        fire_pos = [r for r in records if r["category"] == "Легковоспламеняющиеся" and r["label"] == 1]
        if not fire_pos or args.fire_positive_oversample <= 1:
            return records
        rng = random.Random(args.seed)
        extra = fire_pos * (args.fire_positive_oversample - 1)
        rng.shuffle(extra)
        return records + extra

    def __len__(self):
        return len(self.records)

    def build_prompt(self, record) -> tuple[str, list[Image.Image]]:
        description = str(record["description"])[: self.args.desc_chars]
        user_text = (
            f"Категория проверки: {record['category']}.\n"
            f"Правило: {RULES.get(record['category'], 'правила площадки')}.\n"
            f"Название: {record['name']}\n"
            f"Описание: {description}\n"
            "Относится ли товар к категории? Ответь одним словом: да или нет."
        )
        images = pick_images(list(record["image_paths"]), self.root, self.args.max_images)
        content = [{"type": "text", "text": user_text}]
        for _ in images:
            content.insert(0, {"type": "image"})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return text, images

    def __getitem__(self, index):
        record = self.records[index]
        prompt, images = self.build_prompt(record)
        answer = ("да" if int(record["label"]) == 1 else "нет") if self.include_answer else ""
        return {"record": record, "prompt": prompt, "answer": answer, "images": images}


def make_collate(processor, tokenizer, yes_ids, no_ids):
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        texts = [item["prompt"] + item["answer"] for item in batch]
        images = [item["images"] for item in batch]
        inputs = processor(
            text=texts,
            images=images if any(images) else None,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        has_answer = [bool(item["answer"]) for item in batch]
        if any(has_answer):
            labels = torch.full_like(inputs["input_ids"], -100)
            for row, needs_label in enumerate(has_answer):
                if not needs_label:
                    continue
                # Паддинг СЛЕВА, значит ответ - всегда ПОСЛЕДНЯЯ позиция строки,
                # одна и та же для всех строк батча.
                #
                # Прежний расчёт отсчитывал индексы от начала (full_len - answer_len)
                # и попадал в хвост только у самой длинной строки батча. При батче 16
                # остальные 15 строк размечались ВНУТРИ паддинга: модель училась
                # предсказывать <|endoftext|> в служебной позиции, а настоящий токен
                # ответа не получал надзора вовсе. Симптом - loss на плато 2-5 вместо
                # 0.1-0.5 и F1 на уровне few-shot, будто дообучения не было.
                assert tokenizer.decode(inputs["input_ids"][row, -1]).strip() == batch[row]["answer"].strip(), (
                    "последний токен строки не равен ответу - разметка хвоста "
                    "больше не верна (ответ перестал быть однотокенным?)"
                )
                labels[row, -1] = inputs["input_ids"][row, -1]
            inputs["labels"] = labels
        meta = {
            "ids": [int(item["record"]["id"]) for item in batch],
            "labels": [int(item["record"]["label"]) for item in batch],
        }
        return inputs, meta

    return collate


def resolve_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    def first_ids(words):
        ids = set()
        for word in words:
            encoded = tokenizer(word, add_special_tokens=False)["input_ids"]
            if encoded:
                ids.add(encoded[0])
        return sorted(ids)

    return first_ids(YES_TOKENS), first_ids(NO_TOKENS)


@torch.no_grad()
def evaluate(model, dataset, collate, device, yes_ids, no_ids, batch_size=8):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)
    model.eval()
    ids, truths, probs = [], [], []
    for inputs, meta in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :].float()
        mask = torch.full_like(logits, -1e9)
        allowed = torch.tensor(yes_ids + no_ids, device=logits.device)
        mask[:, allowed] = logits[:, allowed]
        p_yes = mask.softmax(dim=-1)[:, torch.tensor(yes_ids, device=logits.device)].sum(dim=-1)
        ids.extend(meta["ids"])
        truths.extend(meta["labels"])
        probs.extend(p_yes.cpu().tolist())
    model.train()
    return ids, np.asarray(truths), np.asarray(probs)


def competition_report(truths, preds, categories):
    from sklearn.metrics import precision_recall_fscore_support

    rows = {}
    for category in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == category]
        y = truths[idx]
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            pred = (preds[idx] >= t).astype(int)
            if pred.sum() == 0:
                continue
            _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        rows[category] = (best_f1, best_t)
    mean = float(np.mean([v[0] for v in rows.values()])) if rows else 0.0
    return rows, mean


def main():
    global RULES
    args = parse_args()
    if args.rules == "v2":
        RULES = RULES_V2
    # сид управляет инициализацией LoRA и порядком батчей - для сид-супа
    # (усреднение весов адаптеров разных сидов гасит дисперсию обучения)
    import numpy as _np
    random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rules_version.txt").write_text(args.rules, encoding="utf-8")
    pack_dir = Path(args.pack)
    images_root = Path(args.images)

    try:
        from src.runlog import start as _wb_start, log as _wb_log, finish as _wb_finish
    except Exception:
        _wb_start = lambda *a, **k: None
        _wb_log = lambda *a, **k: None
        _wb_finish = lambda *a, **k: None
    _wb_start(
        f"lora-{'all' if args.train_all else 'fold' + str(args.fold)}",
        config=vars(args), tags=["lora"],
    )

    records = dedup(load_records(pack_dir))
    if args.category:
        records = [r for r in records if str(r.get("category", "")) == args.category]
        print(f"фильтр категории «{args.category}»: {len(records)} товаров")
    print(f"товаров после дедупа: {len(records)}")

    from sklearn.model_selection import StratifiedGroupKFold
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    tokenizer = processor.tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ОБУЧЕНИЕ: только bf16, даже если он программный (T4). В цикле нет ни
    # GradScaler, ни autocast, а у fp16 узкий диапазон - активации уходят
    # в inf и loss становится nan с первых шагов (проверено на T4: nan на
    # шаге 10 из 485). bf16 медленнее, но диапазон у него как у fp32.
    # Для ИНФЕРЕНСА обратное верно - там fp16 быстрее и безопасен.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if not torch.cuda.is_bf16_supported():
        print("bf16 недоступен - обучаемся в fp32 (медленно, но без nan)")
    # На двух картах раскладываем слои по обеим: 29 ГБ вместо 14.5 позволяют
    # держать обе картинки в полном разрешении. Скорости это не добавляет
    # (слои считаются по очереди), но снимает ограничение, из-за которого
    # визуальный канал пришлось бы мерить в урезанном виде.
    # Перемещение тензоров между картами accelerate берёт на себя.
    multi_gpu = torch.cuda.device_count() > 1 and not args.single_gpu
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto" if multi_gpu else None,
    )
    if multi_gpu:
        print(f"модель разложена на {torch.cuda.device_count()} карты")
    else:
        model = model.to(device)
    model.config.use_cache = False

    if PEFT:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=(
                ["q_proj", "k_proj", "v_proj", "o_proj"]
                if args.lora_targets == "attn"
                else ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
            ),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Активации 36 слоёв на ~1700 токенов (две картинки дают около тысячи
    # vision-токенов) не помещаются в 16 ГБ. Чекпоинтинг их не хранит, а
    # пересчитывает на обратном проходе: память падает в разы, скорость
    # теряет около трети. enable_input_require_grads обязателен - иначе у
    # замороженного входа нет градиента и чекпоинтинг рвёт граф.
    if args.grad_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("gradient checkpointing включён")
    else:
        print("peft недоступен - будет полное обучение (не рекомендуется на T4)")

    yes_ids, no_ids = resolve_yes_no_ids(tokenizer)
    print(f"yes_ids={yes_ids} no_ids={no_ids}")

    if args.train_all:
        train_records, val_records = records, []
    else:
        y = np.asarray([r["label"] for r in records])
        groups = np.asarray([r["family"] for r in records])
        cv = StratifiedGroupKFold(args.folds_total, shuffle=True, random_state=42)
        folds = list(cv.split(records, y, groups))
        train_idx, val_idx = folds[args.fold]
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

    train_ds = QcDataset(train_records, processor, images_root, args, training=True)
    collate = make_collate(processor, tokenizer, yes_ids, no_ids)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.03
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=collate, shuffle=True,
        num_workers=2, drop_last=True,
    )

    model.train()
    step = 0
    done = False
    accumulated = 0
    for epoch in range(int(np.ceil(args.epochs))):
        for inputs, _meta in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Метка одна на строку - последняя позиция (см. коллатор). Штатный
            # loss заставил бы lm_head посчитать логиты по ВСЕМ позициям: при
            # словаре 248k и батче 16 это тензор порядка 16 ГБ и OutOfMemory.
            # Просим ДВЕ последние позиции: из-за причинного сдвига логиты
            # позиции -2 предсказывают токен на -1, то есть ответ.
            labels = inputs.pop("labels")
            outputs = model(**inputs, logits_to_keep=2)
            loss = F.cross_entropy(
                outputs.logits[:, 0, :].float(), labels[:, -1]
            ) / args.grad_accum
            loss.backward()
            accumulated += 1
            if accumulated % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    current = loss.item() * args.grad_accum
                    print(f"epoch {epoch} step {step}/{total_steps} loss {current:.4f}", flush=True)
                    _wb_log({"loss": current, "epoch": epoch}, step=step)
                if step >= total_steps:
                    done = True
                    break
        if done:
            break

    adapter_dir = out_dir / ("adapter_all" if args.train_all else f"adapter_fold{args.fold}")
    model.save_pretrained(adapter_dir)
    (adapter_dir / "rules_version.txt").write_text(args.rules, encoding="utf-8")
    print(f"адаптер сохранён: {adapter_dir} (правила {args.rules})")

    if val_records:
        val_ds = QcDataset(
            val_records, processor, images_root, args, training=False, include_answer=False
        )
        ids, truths, probs = evaluate(model, val_ds, collate, device, yes_ids, no_ids)
        categories = [r["category"] for r in val_records]
        rows, mean = competition_report(np.asarray(truths), probs, categories)
        for category, (f1, t) in rows.items():
            print(f"{category}: F1={f1:.4f} (порог {t:.2f})")
        print(f"СРЕДНЯЯ МЕТРИКА fold{args.fold}: {mean:.4f}")
        _wb_finish({"mean_f1": float(mean),
                    **{f"f1_{c}": float(v[0]) for c, v in rows.items()}})
        import csv

        with (out_dir / f"oof_fold{args.fold}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink)
            writer.writerow(["id", "prob"])
            writer.writerows(zip(ids, probs))
        print(f"OOF предсказания: {out_dir / f'fold{args.fold}.csv'}")


if __name__ == "__main__":
    main()
