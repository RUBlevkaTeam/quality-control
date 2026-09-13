"""Выбор вычислительного устройства и связанные с ним мелочи.

В проверяющем контейнере стоит H100, поэтому боевой путь - всегда cuda.
MPS (GPU чипов Apple) нужен, чтобы те же модули можно было прогнать локально
на макбуке: отладить пайплайн, померить промпты, посмотреть на реальные
эмбеддинги без аренды сервера.

Логика собрана здесь, чтобы cuda/mps/cpu не расползались по модулям
несогласованными ветками.
"""

from __future__ import annotations

import os
import sys

import torch

# Непокрытые в MPS операции тихо уезжают на CPU вместо NotImplementedError.
# Ставить надо до первого обращения к бэкенду, поэтому - на импорте модуля.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _log(message: str) -> None:
    print(f"[device] {message}", file=sys.stderr, flush=True)


def select_device() -> str:
    """cuda -> mps -> cpu. Приоритет cuda: это боевой путь в контейнере."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# fp16 на MPS считается нестабильно (ядра рассчитаны на fp32/bf16), а на CPU
# половинная точность просто медленная. Поэтому дома работаем в float32.
#
# На cuda предпочитаем bf16: Qwen обучен в нём, и у fp16 узкий диапазон -
# активации переполняются в inf, дальше predict_proba падает на NaN. Но bf16
# есть только с Ampere; на Turing (T4 у Kaggle) остаётся fp16.
def select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


# device_map="auto" тянет accelerate и на маке раскладывает модель странно;
# локально проще положить модель целиком на выбранное устройство.
def model_placement(device: str) -> dict:
    if device == "cuda":
        return {"device_map": "auto"}
    return {}


def free_memory(device: str | None = None) -> None:
    """Отдать кэш аллокатора. Без этого повторная попытка после OOM почти
    гарантированно упирается в тот же OOM."""
    import gc

    gc.collect()
    if device is None:
        device = select_device()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def out_of_memory_errors() -> tuple:
    """Классы OOM-ошибок для except. На MPS отдельного типа нет - там OOM
    прилетает обычным RuntimeError, который мы и так ловим выше по стеку."""
    errors = [torch.cuda.OutOfMemoryError]
    if hasattr(torch, "OutOfMemoryError"):
        errors.append(torch.OutOfMemoryError)
    return tuple(set(errors))


def describe(device: str) -> str:
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"cuda ({name}, {total:.0f} ГБ)"
    if device == "mps":
        return "mps (Apple GPU, локальный прогон)"
    return "cpu"
