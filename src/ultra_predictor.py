"""Runtime orchestration for the ultra text/retrieval/VLM cascade."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np

from src.ultra_features import (
    BAD_CATEGORY,
    FLAMMABLE_CATEGORY,
    evidence_flags,
)
from src.ultra_model import (
    ULTRA_ARTIFACT_FORMAT_VERSION,
    ULTRA_FEATURE_SCHEMA,
    apply_empirical_absence_veto,
    combine_text_and_retrieval,
    predict_text_probabilities,
)
from src.ultra_retrieval import hash_product_images, query_retrieval


def load_ultra_artifact(path: str | Path) -> dict:
    artifact = joblib.load(Path(path))
    validate_ultra_artifact(artifact)
    return artifact


def validate_ultra_artifact(artifact: Mapping[str, object]) -> None:
    """Fail at load/build time when source and trained feature schemas drift."""

    if artifact.get("format_version") != ULTRA_ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported ultra artifact format")
    if artifact.get("feature_schema") != ULTRA_FEATURE_SCHEMA:
        raise ValueError("ultra artifact feature schema does not match runtime")
    text_models = artifact.get("text_models")
    if not isinstance(text_models, Mapping):
        raise ValueError("ultra artifact has no text_models mapping")
    required_model_keys = {
        "word_vectorizer",
        "char_vectorizer",
        "title_vectorizer",
        "classifier",
        "rule_weight",
        "threshold",
        "retrieval_policy",
    }
    for category in (BAD_CATEGORY, FLAMMABLE_CATEGORY):
        model = text_models.get(category)
        if not isinstance(model, Mapping):
            raise ValueError(f"ultra artifact has no model for {category!r}")
        missing = sorted(required_model_keys - set(model))
        if missing:
            raise ValueError(f"ultra model {category!r} misses keys: {missing}")
        threshold = float(model["threshold"])
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"invalid threshold for {category!r}: {threshold}")
    retrieval = artifact.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise ValueError("ultra artifact has no retrieval index")
    missing_retrieval = sorted({"labels", "name", "description", "full", "image"} - set(retrieval))
    if missing_retrieval:
        raise ValueError(f"ultra retrieval index misses keys: {missing_retrieval}")


def _hash_rows(image_paths: Sequence[Sequence[str]]) -> list[tuple[bytes, ...]]:
    # Four workers outperformed wider pools for sequential JPEG files on the
    # reference data and avoids competing with BLAS/transformer CPU threads.
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(hash_product_images, image_paths))


def predict_fast(artifact: Mapping[str, object], dataframe) -> dict[str, object]:
    """Predict with sparse text, exact family memory and empirical vetoes."""

    count = len(dataframe)
    raw_text_probabilities = np.zeros(count, dtype=np.float64)
    final_probabilities = np.zeros(count, dtype=np.float64)
    predictions = np.zeros(count, dtype=np.int8)
    thresholds = np.full(count, 0.5, dtype=np.float64)
    reasons = np.asarray(["text_rules"] * count, dtype=object)
    flags = [
        evidence_flags(row.get("name", ""), row.get("description", ""))
        for _, row in dataframe.iterrows()
    ]
    image_hashes = _hash_rows(dataframe["image_paths"].tolist())
    retrieval_evidence: list[dict | None] = [None] * count

    for position, (_, row) in enumerate(dataframe.iterrows()):
        retrieval_evidence[position] = query_retrieval(
            artifact["retrieval"],
            category=row.get("category", ""),
            name=row.get("name", ""),
            description=row.get("description", ""),
            image_hashes=image_hashes[position],
        )

    for category, model in artifact["text_models"].items():
        mask = dataframe["category"].astype(str).eq(category).to_numpy()
        positions = np.flatnonzero(mask)
        if not len(positions):
            continue
        subset = dataframe.iloc[positions]
        text_probs = predict_text_probabilities(
            model,
            subset["name"].tolist(),
            subset["description"].tolist(),
            subset["category"].tolist(),
        )
        evidence = [retrieval_evidence[position] for position in positions]
        policy = model.get("retrieval_policy", {"mode": "text", "weight": 0.0})
        combined = combine_text_and_retrieval(
            text_probs,
            evidence,
            mode=str(policy.get("mode", "text")),
            weight=float(policy.get("weight", 0.0)),
        )
        threshold = float(model["threshold"])
        raw_text_probabilities[positions] = text_probs
        final_probabilities[positions] = combined
        thresholds[positions] = threshold
        predictions[positions] = (combined >= threshold).astype(np.int8)
        for position, item in zip(positions, evidence):
            if item.get("hard_label") is not None and policy.get("mode") in {"hard", "hybrid"}:
                reasons[position] = str(item.get("hard_reason") or "exact_retrieval")
            elif str(policy.get("mode", "")).startswith("tier_a") and item.get("tier_a_label") is not None:
                mode = str(policy.get("mode"))
                label = int(item["tier_a_label"])
                if mode == "tier_a" or mode == "tier_a_negative" and label == 0 or mode == "tier_a_positive" and label == 1:
                    reasons[position] = "tier_a_exact_retrieval"

    final_probabilities, veto_mask = apply_empirical_absence_veto(
        final_probabilities,
        dataframe["category"].tolist(),
        flags,
    )
    predictions = (final_probabilities >= thresholds).astype(np.int8)
    for position in np.flatnonzero(veto_mask):
        reasons[position] = "fuel_absent_veto"

    return {
        "raw_text_probabilities": raw_text_probabilities,
        "probabilities": final_probabilities,
        "predictions": predictions,
        "thresholds": thresholds,
        "reasons": reasons,
        "flags": flags,
        "retrieval": retrieval_evidence,
        "image_hashes": image_hashes,
    }


def select_vlm_candidates(dataframe, result: Mapping[str, object]) -> list[int]:
    """Route only ambiguous, non-overridden products through Qwen embeddings."""

    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    thresholds = np.asarray(result["thresholds"], dtype=np.float64)
    reasons = np.asarray(result["reasons"], dtype=object)
    eligible = []
    for position, (_, row) in enumerate(dataframe.iterrows()):
        if reasons[position] != "text_rules":
            continue
        category = str(row.get("category", ""))
        band = 0.10 if category == BAD_CATEGORY else 0.14
        distance = abs(probabilities[position] - thresholds[position])
        if distance <= band:
            eligible.append((distance, position))

    max_count = max(1, int(np.ceil(len(dataframe) * 0.18))) if len(dataframe) else 0
    selected = [position for _, position in sorted(eligible)[:max_count]]
    if not selected and len(dataframe):
        # Exercise the permitted VLM path even on tiny check batches, but the
        # high-confidence rescue gates below still prevent arbitrary flips.
        distances = np.abs(probabilities - thresholds)
        selected = [int(np.argmin(distances))]
    return selected


def apply_vlm_tiebreak(
    dataframe,
    result: dict[str, object],
    candidate_positions: Sequence[int],
    vlm_probabilities: Sequence[float],
) -> None:
    """Use very high-confidence VLM output only as a positive visual rescue."""

    if len(candidate_positions) != len(vlm_probabilities):
        raise ValueError("candidate positions and VLM probabilities must align")
    predictions = result["predictions"]
    probabilities = result["probabilities"]
    thresholds = result["thresholds"]
    reasons = result["reasons"]
    flags = result["flags"]
    retrieval = result["retrieval"]

    for position, vlm_probability in zip(candidate_positions, vlm_probabilities):
        if int(predictions[position]) == 1:
            continue
        category = str(dataframe.iloc[position].get("category", ""))
        item_flags = flags[position]
        if category == BAD_CATEGORY:
            strong_negative = bool(
                item_flags.get("bach_flower")
                or item_flags.get("empty_supplement_container")
                or (item_flags.get("not_bad") and item_flags.get("sports"))
            )
            if float(vlm_probability) >= 0.95 and not strong_negative:
                predictions[position] = 1
                probabilities[position] = max(float(probabilities[position]), thresholds[position])
                reasons[position] = "vlm_visual_rescue"
        elif category == FLAMMABLE_CATEGORY:
            retrieval_positive = bool(
                retrieval[position].get("posterior", 0.5) >= 0.75
                and retrieval[position].get("purity", 0.0) >= 0.8
            )
            near_text_boundary = bool(
                result["raw_text_probabilities"][position] >= thresholds[position] * 0.90
            )
            if (
                float(vlm_probability) >= 0.98
                and not item_flags.get("empirical_absence_veto")
                and (
                    item_flags.get("strict_positive_family")
                    or retrieval_positive
                    or near_text_boundary
                )
            ):
                predictions[position] = 1
                probabilities[position] = max(float(probabilities[position]), thresholds[position])
                reasons[position] = "vlm_visual_rescue"
