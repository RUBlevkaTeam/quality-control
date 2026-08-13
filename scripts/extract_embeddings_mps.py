"""Extract and cache Qwen multimodal embeddings using Apple MPS."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.constants import DEFAULT_PIXEL_PRESET, PIXEL_PRESETS
from src.utils_data_prep import prepare_dataframe
from src.utils_embed_cuda import embed_data_cuda

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "content/content_ecup/data.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Local Qwen3-VL-Embedding-2B directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "cache/qwen3_vl_mps.npy",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--pixel-preset",
        choices=tuple(PIXEL_PRESETS),
        default=DEFAULT_PIXEL_PRESET,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Extract only the first N rows for a smoke test",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Install the submission extra with `uv sync --extra submission` "
            "and verify `uv run python -c \"import torch; print(torch.backends.mps.is_available())\"`."
        )
    if not args.model.is_dir():
        raise FileNotFoundError(f"embedding model directory not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"data CSV not found: {args.data}")

    prepared = prepare_dataframe(args.data, args.data.parent / "images")
    if args.limit is not None:
        prepared = prepared.head(args.limit)
    started = datetime.now(timezone.utc)
    print(
        f"Extracting {len(prepared):,} rows on MPS; batch={args.batch_size}; "
        f"pixels={args.pixel_preset}"
    )
    embeddings = embed_data_cuda(
        str(args.model),
        prepared,
        max_pixels=PIXEL_PRESETS[args.pixel_preset],
        batch_size=args.batch_size,
        requested_device="mps",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    pd.DataFrame({"row": np.arange(len(prepared)), "id": prepared["id"]}).to_csv(
        args.output.with_suffix(".ids.csv"), index=False
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "data": str(args.data.resolve()),
        "model": str(args.model.resolve()),
        "rows": len(prepared),
        "dimensions": embeddings.shape[1] if embeddings.ndim == 2 else None,
        "batch_size": args.batch_size,
        "pixel_preset": args.pixel_preset,
        "device": "mps",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved embeddings {embeddings.shape} to {args.output}")


if __name__ == "__main__":
    main()
