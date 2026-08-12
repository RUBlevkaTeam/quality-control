"""Upload an existing local experiment report to the WandB website."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.wandb_tracking import WandbTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "ecup-quality-control"),
    )
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--model", type=Path, help="Matching model artifact, if available")
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()

    if not args.summary.is_file():
        parser.error(f"summary does not exist: {args.summary}")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    required = {"experiment_id", "metrics_at_0_5", "metrics_tuned", "thresholds"}
    missing = required - set(summary)
    if missing:
        raise ValueError(f"summary is missing fields: {sorted(missing)}")

    oof_path = Path(summary["oof_path"])
    error_count = 0
    if oof_path.is_file():
        oof = pd.read_csv(oof_path)
        if "is_error" in oof:
            error_count = int(oof["is_error"].astype(bool).sum())
        elif {"label", "prediction"}.issubset(oof.columns):
            error_count = int((oof["label"] != oof["prediction"]).sum())

    experiment_id = str(summary["experiment_id"])
    tracker = WandbTracker.start(
        enabled=True,
        project=args.project,
        entity=args.entity,
        mode=args.mode,
        name=experiment_id,
        group="imported-local-reports",
        tags=["imported", "oof", *args.tag],
        config={
            "experiment_id": experiment_id,
            "source": "existing-local-report",
            "rows": summary.get("rows"),
            "n_splits": summary.get("n_splits"),
            **summary.get("config", {}),
        },
    )
    tracker.log_final(
        metrics_at_05=summary["metrics_at_0_5"],
        metrics_tuned=summary["metrics_tuned"],
        thresholds=summary["thresholds"],
        runtime_seconds=float(summary.get("runtime_seconds", 0.0)),
        error_count=error_count,
    )
    report_dir = args.summary.parent
    tracker.log_artifacts(
        experiment_id=experiment_id,
        report_files=[
            args.summary,
            oof_path,
            report_dir / "folds.csv",
            report_dir / "error_analysis.csv",
        ],
        model_path=args.model,
    )
    run_url = tracker.run.url
    tracker.finish()
    print(f"Uploaded {experiment_id} to {run_url or 'the configured offline run'}")


if __name__ == "__main__":
    main()

