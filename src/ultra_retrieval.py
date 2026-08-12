"""Exact, precision-first product-family retrieval.

The index keeps posting lists rather than connected components.  This avoids
transitive label contamination through shared marketplace banners while still
letting one test product collect and deduplicate all matching train products.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.ultra_features import normalize_text, normalized_full_text, normalized_name


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths_for_id(images_root: str | Path, product_id: object) -> list[Path]:
    directory = Path(images_root) / str(product_id)
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> bytes:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.digest()


def hash_product_images(paths: Iterable[str | Path]) -> tuple[bytes, ...]:
    """Hash readable images; a bad file is ignored instead of killing inference."""

    hashes: list[bytes] = []
    for path in paths:
        try:
            hashes.append(sha256_file(path))
        except (OSError, ValueError):
            continue
    return tuple(sorted(set(hashes)))


def _text_digest(normalized: str) -> bytes:
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def hash_dataset_images(ids: Sequence[object], images_root: str | Path) -> list[tuple[bytes, ...]]:
    return [hash_product_images(image_paths_for_id(images_root, product_id)) for product_id in ids]


def _freeze_postings(postings: Mapping[object, set[int]]) -> dict[object, tuple[int, ...]]:
    return {key: tuple(sorted(rows)) for key, rows in postings.items()}


def build_retrieval_index(
    names: Sequence[object],
    descriptions: Sequence[object],
    categories: Sequence[object],
    labels: Sequence[int],
    image_hashes: Sequence[Sequence[bytes]],
    row_indices: Sequence[int] | None = None,
) -> dict:
    """Build compact exact-name, exact-text and exact-image posting lists."""

    size = len(names)
    if not (size == len(descriptions) == len(categories) == len(labels) == len(image_hashes)):
        raise ValueError("retrieval inputs must have equal length")
    if row_indices is None:
        row_indices = list(range(size))
    if len(row_indices) != size:
        raise ValueError("row_indices must match retrieval input length")

    names_map: dict[str, dict[bytes, set[int]]] = defaultdict(lambda: defaultdict(set))
    description_map: dict[str, dict[bytes, set[int]]] = defaultdict(lambda: defaultdict(set))
    full_map: dict[str, dict[bytes, set[int]]] = defaultdict(lambda: defaultdict(set))
    image_map: dict[str, dict[bytes, set[int]]] = defaultdict(lambda: defaultdict(set))

    max_index = max(row_indices, default=-1)
    label_array = np.full(max_index + 1, -1, dtype=np.int8)

    for local_i, global_i in enumerate(row_indices):
        category = str(categories[local_i])
        label_array[global_i] = int(labels[local_i])

        name_key = normalized_name(names[local_i])
        description_key = normalize_text(descriptions[local_i])
        full_key = normalized_full_text(names[local_i], descriptions[local_i])
        if name_key:
            names_map[category][_text_digest(name_key)].add(global_i)
        if description_key:
            description_map[category][_text_digest(description_key)].add(global_i)
        if full_key != "\x1f":
            full_map[category][_text_digest(full_key)].add(global_i)
        for digest in set(image_hashes[local_i]):
            image_map[category][digest].add(global_i)

    return {
        "labels": label_array,
        "name": {
            category: _freeze_postings(values) for category, values in names_map.items()
        },
        "description": {
            category: _freeze_postings(values)
            for category, values in description_map.items()
        },
        "full": {
            category: _freeze_postings(values) for category, values in full_map.items()
        },
        "image": {
            category: _freeze_postings(values) for category, values in image_map.items()
        },
    }


def _source_summary(indices: set[int], labels: np.ndarray) -> dict[str, object]:
    valid = [index for index in indices if 0 <= index < len(labels) and labels[index] in (0, 1)]
    n1 = sum(int(labels[index]) for index in valid)
    n0 = len(valid) - n1
    unanimous_label = None
    if valid and (n0 == 0 or n1 == 0):
        unanimous_label = int(n1 > 0)
    return {
        "indices": set(valid),
        "support": len(valid),
        "n0": n0,
        "n1": n1,
        "unanimous_label": unanimous_label,
        "purity": (max(n0, n1) / len(valid)) if valid else 0.0,
    }


def query_retrieval(
    index: Mapping[str, object],
    *,
    category: object,
    name: object,
    description: object,
    image_hashes: Sequence[bytes],
) -> dict[str, object]:
    """Collect deduplicated train neighbours and calibrated evidence metadata."""

    category_key = str(category)
    labels = np.asarray(index["labels"], dtype=np.int8)
    name_key = normalized_name(name)
    description_key = normalize_text(description)
    full_key = normalized_full_text(name, description)

    name_postings = index.get("name", {}).get(category_key, {})
    description_postings = index.get("description", {}).get(category_key, {})
    full_postings = index.get("full", {}).get(category_key, {})
    image_postings = index.get("image", {}).get(category_key, {})

    source_indices: dict[str, set[int]] = {
        "name": set(name_postings.get(_text_digest(name_key), ())) if name_key else set(),
        "description": (
            set(description_postings.get(_text_digest(description_key), ()))
            if description_key
            else set()
        ),
        "full": (
            set(full_postings.get(_text_digest(full_key), ()))
            if full_key != "\x1f"
            else set()
        ),
        "image": set(),
    }
    matched_image_count = 0
    for digest in set(image_hashes):
        postings = image_postings.get(digest, ())
        if postings:
            matched_image_count += 1
            source_indices["image"].update(postings)

    summaries = {
        source: _source_summary(indices, labels) for source, indices in source_indices.items()
    }
    union_indices = set().union(*(summary["indices"] for summary in summaries.values()))
    union_summary = _source_summary(union_indices, labels)

    active = [source for source, summary in summaries.items() if summary["support"]]
    source_labels = [summaries[source]["unanimous_label"] for source in active]
    all_sources_unanimous = bool(active) and all(label is not None for label in source_labels)
    sources_agree = all_sources_unanimous and len(set(source_labels)) == 1

    independent_sources = ["name", "description", "image"]
    tier_a_active = [source for source in independent_sources if summaries[source]["support"]]
    tier_a_labels = [summaries[source]["unanimous_label"] for source in tier_a_active]
    tier_a_label = None
    if (
        len(tier_a_active) >= 2
        and all(label is not None for label in tier_a_labels)
        and len(set(tier_a_labels)) == 1
    ):
        tier_a_label = int(tier_a_labels[0])

    # Reliability is intentionally conservative.  The caller may still use the
    # posterior as a soft feature when hard_label is None.
    hard_label = None
    hard_reason = None
    if union_summary["unanimous_label"] is not None:
        candidate = int(union_summary["unanimous_label"])
        if summaries["full"]["support"]:
            hard_label = candidate
            hard_reason = "exact_full_text"
        elif category_key.endswith("Легковоспламеняющиеся") and summaries["description"]["support"]:
            hard_label = candidate
            hard_reason = "exact_description_flammable"
        elif summaries["image"]["support"] and summaries["name"]["support"] and sources_agree:
            hard_label = candidate
            hard_reason = "exact_name_and_image"
        elif summaries["image"]["support"] >= 2 or matched_image_count >= 2:
            hard_label = candidate
            hard_reason = "multiple_exact_images"
        elif category_key.endswith("Легковоспламеняющиеся") and summaries["image"]["support"]:
            hard_label = candidate
            hard_reason = "exact_image_flammable"
        elif category_key.endswith("Легковоспламеняющиеся") and summaries["name"]["support"] >= 2:
            hard_label = candidate
            hard_reason = "repeated_exact_name_flammable"

    # Beta smoothing keeps a one-neighbour match away from mathematical 0/1.
    prior = 0.5
    support = int(union_summary["support"])
    posterior = (int(union_summary["n1"]) + prior) / (support + 2 * prior) if support else 0.5

    return {
        "hard_label": hard_label,
        "hard_reason": hard_reason,
        "posterior": float(posterior),
        "support": support,
        "purity": float(union_summary["purity"]),
        "matched_sources": tuple(active),
        "matched_image_count": matched_image_count,
        "sources_agree": bool(sources_agree),
        "tier_a_label": tier_a_label,
        "source_summaries": summaries,
    }
