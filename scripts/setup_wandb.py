"""Authenticate WandB or verify the local offline tracking setup."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "ecup-quality-control"),
    )
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Create a local smoke run without authentication or network",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is not installed; run `uv sync --extra tracking` first"
        ) from error

    if not args.offline:
        # The SDK reads the key interactively or from WANDB_API_KEY. The key is
        # never accepted as our CLI argument and is never written to the repo.
        print("Open https://wandb.ai/authorize in your browser to create/copy the API key.")
        print("https://api.wandb.ai is the SDK endpoint and returns 404 in a browser.")
        wandb.login(host="https://api.wandb.ai", verify=True)

    with wandb.init(
        project=args.project,
        entity=args.entity,
        mode="offline" if args.offline else "online",
        name="wandb-setup-check",
        job_type="setup",
        config={"purpose": "verify quality-control experiment tracking"},
    ) as run:
        run.log({"setup/ok": 1})
        run_url = run.url

    if args.offline:
        print("WandB offline smoke run completed.")
        print("Use `wandb sync wandb/offline-run-*` to upload it later.")
    else:
        print(f"WandB is ready: {run_url}")


if __name__ == "__main__":
    main()
