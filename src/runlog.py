"""Единая точка логирования экспериментов в W&B.

Безопасный no-op: без установленного wandb или без логина ничего не падает -
обучение важнее телеметрии. Оффлайн-режим включается WANDB_MODE=offline.

Использование:
    from src.runlog import start, log, finish
    start("lora-strict-fold0", config={...})
    log({"loss": 0.12, "step": 10})
    finish({"f1_fire": 0.71})
"""
from __future__ import annotations

import os
import sys

_run = None
PROJECT = os.environ.get("WANDB_PROJECT", "ecup-quality")


def _say(msg: str) -> None:
    print(f"[runlog] {msg}", file=sys.stderr, flush=True)


def start(name: str, config: dict | None = None, tags: list | None = None):
    global _run
    try:
        import wandb

        _run = wandb.init(project=PROJECT, name=name, config=config or {},
                          tags=tags or [], reinit=True)
        _say(f"W&B: {name} -> {getattr(_run, 'url', 'offline')}")
    except Exception as exc:
        _run = None
        _say(f"W&B недоступен ({type(exc).__name__}), логирую только в stdout")
    return _run


def log(metrics: dict, step: int | None = None) -> None:
    if _run is not None:
        try:
            _run.log(metrics, step=step)
        except Exception:
            pass


def finish(summary: dict | None = None) -> None:
    global _run
    if _run is not None:
        try:
            for k, v in (summary or {}).items():
                _run.summary[k] = v
            _run.finish()
        except Exception:
            pass
    _run = None
