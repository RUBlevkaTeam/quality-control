#!/usr/bin/env python3
"""Build the platform-neutral train image retrieval artifact."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.image_retrieval import (
    build_retrieval_index,
    fingerprint_products,
    image_paths_for_id,
    save_retrieval_index,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "content/content_ecup/images")
    parser.add_argument("--output", type=Path, default=ROOT / "image_retrieval.npz")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataframe = pd.read_csv(args.data)
    paths = [image_paths_for_id(args.images, value) for value in dataframe["id"]]
    started = time.perf_counter()
    fingerprints = fingerprint_products(paths, workers=args.workers)
    elapsed = time.perf_counter() - started
    index = build_retrieval_index(dataframe, fingerprints)
    save_retrieval_index(args.output, index)
    print(
        f"products={len(dataframe)} images={len(index['dhashes'])} "
        f"backend_artifact={args.output} size={args.output.stat().st_size} "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
