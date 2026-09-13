"""LoRA-классификатор Qwen3.5-4B (vision) на инференсе платформы.

Пара к scripts/train_lora_qc.py: тот же промпт, те же лимиты картинок и
описания - рассинхрон train/inference здесь стоит метрики. Вероятность =
softmax по логитам первых токенов {да, нет} на позиции после промпта,
поэтому никакой генерации текста и парсинга.

Все сбои деградируют до пустого списка вероятностей: упавший прогон дороже.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence

from PIL import Image

DESC_CHARS = 700
MAX_IMAGES = 2

SYSTEM_PROMPT = (
    "Ты модератор маркетплейса. По карточке товара и фотографиям решаешь, "
    "относится ли товар к указанной категории. Отвечай одним словом."
)

# Версии правил в промпте: v1 - сжатый пересказ, v2 - дословный текст условий.
# Какую версию использовать, решает маркер rules_version.txt в папке адаптера
# (его пишет train_lora_qc) - промпт инференса всегда совпадает с обучением.
# Литерал RULES_V2 обязан побайтово совпадать с scripts/train_lora_qc.py.
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


def _log(message: str) -> None:
    print(f"[lora] {message}", file=sys.stderr, flush=True)


def load_blend_config(path: Path) -> dict:
    """{'weights': {'бад': 0.6, ...}, 'thresholds': {...}} либо {}."""

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _pick_images(paths, root_fallback=None) -> List[Image.Image]:
    # площадь читается из заголовка без декодирования пикселей; полное
    # декодирование - только для выбранных MAX_IMAGES (до правки декодировали
    # все 6 и выбрасывали 4 - минуты CPU на полном прогоне)
    sizes = []
    for rel in list(paths)[:6]:
        try:
            with Image.open(rel) as probe:
                sizes.append((probe.width * probe.height, rel))
        except OSError:
            continue
    sizes.sort(key=lambda pair: pair[0], reverse=True)
    chosen: List[Image.Image] = []
    for _, rel in sizes:
        if len(chosen) >= MAX_IMAGES:
            break
        try:
            with Image.open(rel) as probe:
                chosen.append(probe.convert("RGB").copy())
        except OSError:
            continue
    return chosen


def _attach_adapter_hooks(base, adapter_path, torch) -> int:
    """Запасной путь без peft: LoRA-поправка на лету, как считает сам peft.

    Прежний вариант вливал W += (alpha/r)·B@A в bf16-веса, но дельта при
    этом дважды округляется до bf16 и частично теряется: verify показал
    расхождение вероятностей с peft до 0.03. Поэтому веса не трогаем:
    A/B висят на модуле буферами, forward-хук добавляет scale·B(A(x))
    с теми же типами и порядком округлений, что peft
    (x -> dtype A, сложение с промоцией, каст обратно в dtype выхода).

    Ключи в adapter_model.safetensors выглядят как
    base_model.model.<путь.модуля>.lora_A.weight - срезаем обёртку peft и
    находим модуль в базовой модели по этому пути.
    """
    import json as _json

    from safetensors.torch import load_file

    directory = Path(adapter_path)
    config = _json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    scale = float(config["lora_alpha"]) / float(config["r"])
    state = load_file(str(directory / "adapter_model.safetensors"))

    pairs: dict = {}
    for key, tensor in state.items():
        if ".lora_A." in key:
            pairs.setdefault(key.replace(".lora_A.", "."), {})["A"] = tensor
        elif ".lora_B." in key:
            pairs.setdefault(key.replace(".lora_B.", "."), {})["B"] = tensor

    prefix = "base_model.model."
    attached = 0
    for key, parts in pairs.items():
        if "A" not in parts or "B" not in parts:
            continue
        path = key[len(prefix):] if key.startswith(prefix) else key
        if path.endswith(".weight"):
            path = path[: -len(".weight")]
        module = base.get_submodule(path)
        # буферы переезжают на device вместе с model.to(...), dtype сохраняется
        module.register_buffer("qc_lora_a", parts["A"].clone(), persistent=False)
        module.register_buffer("qc_lora_b", parts["B"].clone(), persistent=False)

        def hook(mod, args, output, _scale=scale):
            x = args[0]
            update = torch.nn.functional.linear(
                torch.nn.functional.linear(x.to(mod.qc_lora_a.dtype), mod.qc_lora_a),
                mod.qc_lora_b,
            ) * _scale
            return (output + update).to(output.dtype)

        module.register_forward_hook(hook)
        attached += 1
    if attached == 0:
        raise RuntimeError("в адаптере не нашлось ни одной пары lora_A/lora_B")
    return attached


class LoraClassifier:
    def __init__(self, base_model_path: str, adapter_path: str, max_pixels: int = 512 * 28 * 28):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        # версия правил в промпте = та, на которой обучался адаптер
        try:
            version = (Path(adapter_path) / "rules_version.txt").read_text(
                encoding="utf-8").strip()
        except OSError:
            version = "v1"
        self.rules = RULES_V2 if version == "v2" else RULES
        _log(f"промпт правил: {version}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.processor = AutoProcessor.from_pretrained(
            base_model_path, local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=max_pixels,
        )
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        base = AutoModelForImageTextToText.from_pretrained(
            base_model_path, dtype=dtype, local_files_only=True, attn_implementation="sdpa"
        )
        # Наличие peft в образе проверить нельзя, а его отсутствие стоило бы
        # всего канала: ImportError откатил бы решение на чистый текст. Поэтому
        # без peft вешаем LoRA-поправку форвард-хуками - та же арифметика,
        # что у peft (вливание в bf16-веса теряло точность, verify ловил 0.03).
        try:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
            _log("адаптер подключён через peft")
        except ImportError:
            attached = _attach_adapter_hooks(base, adapter_path, torch)
            _log(f"peft недоступен - LoRA-хуки на {attached} матрицах")
            self.model = base
        self.model.to(self.device)
        self.model.eval()

        def first_ids(words):
            ids = set()
            for word in words:
                encoded = self.processor.tokenizer(word, add_special_tokens=False)["input_ids"]
                if encoded:
                    ids.add(encoded[0])
            return sorted(ids)

        self.yes_ids = first_ids(("да", " Да", "да."))
        self.no_ids = first_ids(("нет", " Нет", "нет."))
        _log(f"загружен, устройство {self.device}, yes={self.yes_ids} no={self.no_ids}")

    def _prompt(self, row, image_count: int) -> str:
        description = str(row.get("description", ""))[:DESC_CHARS]
        category = str(row.get("category", ""))
        user_text = (
            f"Категория проверки: {category}.\n"
            f"Правило: {self.rules.get(category, 'правила площадки')}.\n"
            f"Название: {row.get('name', '')}\n"
            f"Описание: {description}\n"
            "Относится ли товар к категории? Ответь одним словом: да или нет."
        )
        content = [{"type": "text", "text": user_text}]
        # Плейсхолдеров ровно столько, сколько картинок РЕАЛЬНО загрузилось -
        # как в обучении (build_prompt: for _ in images). Жёсткие два ломали
        # весь батч на товарах с одной картинкой (их 14%): процессор падал на
        # рассинхроне, и все 8 товаров батча получали заглушку 0.5.
        for _ in range(image_count):
            content.insert(0, {"type": "image"})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def predict_proba(self, df, batch_size: int = 8) -> List[float]:
        probs: List[float] = []
        total = len(df)
        for start in range(0, total, batch_size):
            chunk = df.iloc[start : start + batch_size]
            prompts, images_per_row = [], []
            for _, row in chunk.iterrows():
                images = _pick_images(row.get("image_paths") or [])
                images_per_row.append(images)
                prompts.append(self._prompt(row, len(images)))
            try:
                inputs = self.processor(
                    text=prompts,
                    images=images_per_row if any(images_per_row) else None,
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt",
                ).to(self.device)
                with self.torch.no_grad():
                    try:
                        # логиты только последней позиции: полный тензор
                        # (батч x 2048 x 150k словаря) стоил гигабайты VRAM
                        logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :].float()
                    except TypeError:
                        logits = self.model(**inputs).logits[:, -1, :].float()
                # softmax только по «да»/«нет»-токенам вместо клона всего словаря
                allowed = self.torch.tensor(self.yes_ids + self.no_ids, device=logits.device)
                sub = logits[:, allowed].softmax(dim=-1)
                p_yes = sub[:, : len(self.yes_ids)].sum(dim=-1)
                probs.extend(p_yes.tolist())
            except Exception as exc:
                _log(f"батч {start} не отработал ({type(exc).__name__}: {exc})")
                # None = «канал промолчал»: run.py оставит решение текстовой
                # модели с её порогом (0.5 в логит-смеси искажал вердикты)
                probs.extend([None] * len(chunk))
            finally:
                for images in images_per_row:
                    for image in images:
                        image.close()
        return probs
