#!/usr/bin/env python3
"""Leave-one-product-out diagnostics for exact/perceptual image retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.image_retrieval import HammingMatcher, load_retrieval_index, retrieval_decision


def _exact_neighbours(index) -> list[set[int]]:
    labels = index["product_labels"]
    output = [set() for _ in labels]
    keys = index["sha_keys"]
    products = index["sha_products"]
    start = 0
    while start < len(keys):
        end = start + 1
        while end < len(keys) and keys[end] == keys[start]:
            end += 1
        group = set(map(int, products[start:end]))
        for product in group:
            output[product].update(group - {product})
        start = end
    return output


def _perceptual_neighbours(index, matcher, max_distance: int, top_k: int):
    labels = index["product_labels"]
    output = [dict() for _ in labels]
    image_products = index["dhash_products"]
    product_categories = index["product_categories"]
    timings = {}
    for code, name in enumerate(index["category_names"]):
        rows = np.flatnonzero(product_categories[image_products] == code)
        hashes = index["dhashes"][rows]
        products = image_products[rows]
        started = time.perf_counter()
        indices, distances = matcher.topk(
            hashes, hashes, max_distance=max_distance, top_k=top_k
        )
        timings[str(name)] = time.perf_counter() - started
        for query_index, (matches, values) in enumerate(zip(indices, distances)):
            query_product = int(products[query_index])
            for local_index, distance in zip(matches, values):
                if local_index < 0:
                    break
                neighbour = int(products[int(local_index)])
                if neighbour == query_product:
                    continue
                output[query_product][neighbour] = min(
                    int(distance), output[query_product].get(neighbour, 257)
                )
    return output, timings


def _metrics(y_true, y_pred) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int8)
    pred = np.asarray(y_pred, dtype=np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return {
        "calls": int(len(y)),
        "accuracy": float(np.mean(y == pred)) if len(y) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=ROOT / "image_retrieval.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/image_retrieval_loo.json")
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=128)
    args = parser.parse_args()

    index = load_retrieval_index(args.index)
    matcher = HammingMatcher()
    exact = _exact_neighbours(index)
    perceptual, timings = _perceptual_neighbours(
        index, matcher, args.max_distance, args.top_k
    )
    labels = index["product_labels"]
    categories = index["product_categories"]
    names = index["product_name_digest"]
    report = {
        "protocol": "leave-one-product-out direct neighbours; no transitive components",
        "backend": matcher.backend,
        "max_distance": args.max_distance,
        "top_k": args.top_k,
        "native_query_seconds": timings,
        "categories": {},
    }
    for code, category in enumerate(index["category_names"]):
        true = []
        predicted = []
        reasons = defaultdict(int)
        for product in np.flatnonzero(categories == code):
            exact_records = tuple(
                (neighbour, int(labels[neighbour]), None, bool(names[neighbour] == names[product]))
                for neighbour in sorted(exact[product])
            )
            perceptual_records = tuple(
                (
                    neighbour,
                    int(labels[neighbour]),
                    distance,
                    bool(names[neighbour] == names[product]),
                )
                for neighbour, distance in sorted(perceptual[product].items())
            )
            decision, reason = retrieval_decision(
                str(category),
                {
                    "exact_products": exact_records,
                    "perceptual_products": perceptual_records,
                },
            )
            reasons[reason] += 1
            if decision is not None:
                true.append(int(labels[product]))
                predicted.append(decision)
        metrics = _metrics(true, predicted)
        metrics["coverage"] = metrics["calls"] / int(np.sum(categories == code))
        metrics["reasons"] = dict(reasons)
        report["categories"][str(category)] = metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
