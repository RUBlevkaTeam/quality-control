"""Optional Weights & Biases integration for local experiments.

The module imports wandb lazily so the base and submission environments do not
depend on the tracking SDK.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

_CATEGORY_NAMES = {
    "БАД": "bad",
    "Легковоспламеняющиеся": "flammable",
}


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common opt-in WandB CLI flags to an experiment parser."""
    parser.add_argument("--wandb", action="store_true", help="Enable WandB tracking")
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "ecup-quality-control"),
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--wandb-tag",
        action="append",
        default=[],
        dest="wandb_tags",
        help="Repeat the option to attach several tags",
    )
    parser.add_argument(
        "--wandb-log-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload reports and the final model artifact",
    )


def category_metric_name(category: str) -> str:
    """Return a stable ASCII component for a category metric key."""
    if category in _CATEGORY_NAMES:
        return _CATEGORY_NAMES[category]
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(category)).strip("_").lower()
    return normalized or "unknown"


def metrics_payload(
    metrics_at_05: dict[str, object],
    metrics_tuned: dict[str, object],
    thresholds: dict[str, float],
    runtime_seconds: float,
    error_count: int,
) -> dict[str, float | int]:
    """Flatten competition metrics into stable W&B dashboard keys."""
    payload: dict[str, float | int] = {
        "metrics/mean_f1_at_0_5": float(metrics_at_05["mean_f1"]),
        "metrics/mean_f1_tuned": float(metrics_tuned["mean_f1"]),
        "runtime/total_seconds": float(runtime_seconds),
        "errors/count": int(error_count),
    }
    for category, values in metrics_tuned["per_category"].items():
        name = category_metric_name(str(category))
        payload[f"metrics/{name}/f1"] = float(values["f1"])
        payload[f"metrics/{name}/precision"] = float(values["precision"])
        payload[f"metrics/{name}/recall"] = float(values["recall"])
        payload[f"data/{name}/count"] = int(values["count"])
    for category, threshold in thresholds.items():
        payload[f"thresholds/{category_metric_name(str(category))}"] = float(threshold)
    return payload


class WandbTracker:
    """Small adapter that keeps WandB optional for the rest of the project."""

    def __init__(self, run: Any | None = None, wandb_module: Any | None = None):
        self.run = run
        self._wandb = wandb_module

    @property
    def enabled(self) -> bool:
        return self.run is not None

    @classmethod
    def start(
        cls,
        *,
        enabled: bool,
        project: str,
        entity: str | None,
        mode: str,
        name: str,
        config: dict[str, object],
        tags: list[str],
        group: str | None = None,
    ) -> "WandbTracker":
        if not enabled:
            return cls()
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "WandB tracking was requested but wandb is not installed. "
                "Run `uv sync --extra tracking`."
            ) from error

        run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            name=name,
            group=group,
            job_type="cross-validation",
            config=config,
            tags=tags,
            save_code=True,
        )
        return cls(run=run, wandb_module=wandb)

    def log_fold(
        self,
        *,
        fold: int,
        train_rows: int,
        valid_rows: int,
        runtime_seconds: float,
        metrics: dict[str, object],
    ) -> None:
        if not self.enabled:
            return
        payload: dict[str, float | int] = {
            "fold/index": fold,
            "fold/train_rows": train_rows,
            "fold/valid_rows": valid_rows,
            "fold/runtime_seconds": runtime_seconds,
            "fold/mean_f1_at_0_5": float(metrics["mean_f1"]),
        }
        for category, values in metrics["per_category"].items():
            name = category_metric_name(str(category))
            payload[f"fold/{name}/f1_at_0_5"] = float(values["f1"])
            payload[f"fold/{name}/precision_at_0_5"] = float(values["precision"])
            payload[f"fold/{name}/recall_at_0_5"] = float(values["recall"])
        self.run.log(payload, step=fold)

    def log_final(
        self,
        *,
        metrics_at_05: dict[str, object],
        metrics_tuned: dict[str, object],
        thresholds: dict[str, float],
        runtime_seconds: float,
        error_count: int,
    ) -> None:
        if not self.enabled:
            return
        payload = metrics_payload(
            metrics_at_05,
            metrics_tuned,
            thresholds,
            runtime_seconds,
            error_count,
        )
        self.run.log(payload)
        for key, value in payload.items():
            self.run.summary[key] = value

    def log_artifacts(
        self,
        *,
        experiment_id: str,
        report_files: list[Path],
        model_path: Path | None = None,
    ) -> None:
        if not self.enabled:
            return
        report_artifact = self._wandb.Artifact(
            name=f"{experiment_id}-reports", type="evaluation"
        )
        for path in report_files:
            if path.is_file():
                report_artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(report_artifact)

        if model_path is not None and model_path.is_file():
            model_artifact = self._wandb.Artifact(
                name=f"{experiment_id}-model", type="model"
            )
            model_artifact.add_file(str(model_path), name=model_path.name)
            self.run.log_artifact(model_artifact)

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()
