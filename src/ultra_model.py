"""Sparse text classifier used by the ultra submission.

The model deliberately stays small and deterministic: category-specific word
and character TF-IDF, a compact rule bank, and logistic regression.  The
multimodal Qwen baseline is used later as a routed tie-breaker rather than
forcing every decision through a generative model.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.ultra_features import (
    FLAMMABLE_CATEGORY,
    RULE_FEATURE_NAMES,
    build_model_texts,
    rule_feature_matrix,
)


ULTRA_ARTIFACT_FORMAT_VERSION = 2
ULTRA_FEATURE_SCHEMA = {
    "version": 2,
    "text_features": "word-char-title-rules",
    "rule_feature_names": RULE_FEATURE_NAMES,
    "retrieval_text_digest": "blake2b-128",
}


def _make_vectorizers(config: Mapping[str, object]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=int(config.get("word_min_df", 2)),
        max_df=float(config.get("max_df", 0.999)),
        max_features=int(config.get("word_max_features", 120_000)),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=int(config.get("char_min_df", 2)),
        max_df=float(config.get("max_df", 0.999)),
        max_features=int(config.get("char_max_features", 180_000)),
        sublinear_tf=True,
        dtype=np.float32,
    )
    title = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        min_df=int(config.get("title_min_df", 2)),
        max_df=float(config.get("max_df", 0.999)),
        max_features=int(config.get("title_max_features", 70_000)),
        sublinear_tf=True,
        dtype=np.float32,
    )
    return word, char, title


def _stack_features(
    word_matrix,
    char_matrix,
    title_matrix,
    rules: np.ndarray,
    rule_weight: float,
):
    from scipy import sparse

    rule_matrix = sparse.csr_matrix(rules * np.float32(rule_weight), dtype=np.float32)
    return sparse.hstack(
        (word_matrix, char_matrix, title_matrix, rule_matrix),
        format="csr",
        dtype=np.float32,
    )


def fit_text_model(
    names: Sequence[object],
    descriptions: Sequence[object],
    categories: Sequence[object],
    labels: Sequence[int],
    *,
    config: Mapping[str, object] | None = None,
) -> dict:
    from sklearn.linear_model import LogisticRegression

    config = dict(config or {})
    texts = build_model_texts(names, descriptions)
    rules = rule_feature_matrix(list(names), list(descriptions), list(categories))
    normalized_titles = build_model_texts(names, [""] * len(names))
    word, char, title = _make_vectorizers(config)
    word_matrix = word.fit_transform(texts)
    char_matrix = char.fit_transform(texts)
    title_matrix = title.fit_transform(normalized_titles)
    rule_weight = float(config.get("rule_weight", 2.0))
    matrix = _stack_features(word_matrix, char_matrix, title_matrix, rules, rule_weight)

    class_weight = config.get("class_weight", None)
    classifier = LogisticRegression(
        C=float(config.get("C", 3.0)),
        class_weight=class_weight,
        max_iter=int(config.get("max_iter", 1200)),
        solver=str(config.get("solver", "liblinear")),
        random_state=int(config.get("random_state", 2026)),
    )
    classifier.fit(matrix, np.asarray(labels, dtype=np.int8))
    return {
        "word_vectorizer": word,
        "char_vectorizer": char,
        "title_vectorizer": title,
        "classifier": classifier,
        "rule_weight": rule_weight,
        "config": config,
    }


def predict_text_probabilities(
    model: Mapping[str, object],
    names: Sequence[object],
    descriptions: Sequence[object],
    categories: Sequence[object],
) -> np.ndarray:
    if not names:
        return np.empty(0, dtype=np.float64)
    texts = build_model_texts(names, descriptions)
    normalized_titles = build_model_texts(names, [""] * len(names))
    rules = rule_feature_matrix(list(names), list(descriptions), list(categories))
    word_matrix = model["word_vectorizer"].transform(texts)
    char_matrix = model["char_vectorizer"].transform(texts)
    title_matrix = model["title_vectorizer"].transform(normalized_titles)
    matrix = _stack_features(
        word_matrix,
        char_matrix,
        title_matrix,
        rules,
        float(model["rule_weight"]),
    )
    return np.asarray(model["classifier"].predict_proba(matrix)[:, 1], dtype=np.float64)


def best_f1_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> tuple[float, float]:
    """Find the exact OOF threshold at score boundaries, with stable tie-breaking."""

    y = np.asarray(labels, dtype=np.int8)
    probs = np.asarray(probabilities, dtype=np.float64)
    if not len(y):
        return 0.5, 0.0
    if len(y) != len(probs):
        raise ValueError("labels and probabilities must align")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities must be finite")

    # For predictions `score >= threshold`, a decision can only change at a
    # unique observed score.  Sorting once makes this an exact O(n log n)
    # search instead of sampling quantiles or recomputing F1 thousands of times.
    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order]
    sorted_y = y[order]
    cumulative_tp = np.cumsum(sorted_y, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_y, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_probs[:-1] != sorted_probs[1:], True]
    )
    tp = cumulative_tp[group_ends].astype(np.float64)
    fp = cumulative_fp[group_ends].astype(np.float64)
    fn = float(sorted_y.sum()) - tp
    denominator = 2.0 * tp + fp + fn
    scores = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0,
    )
    best_f1 = float(scores.max(initial=0.0))
    tied = np.flatnonzero(np.isclose(scores, best_f1, rtol=0.0, atol=1e-12))
    # A larger threshold is precision-first on an exact F1 tie.
    best_threshold = float(np.max(sorted_probs[group_ends[tied]]))
    return best_threshold, best_f1


def combine_text_and_retrieval(
    text_probabilities: Sequence[float],
    evidence: Sequence[Mapping[str, object]],
    *,
    mode: str,
    weight: float = 1.0,
) -> np.ndarray:
    """Apply the exact same retrieval policy during OOF training and inference."""

    probabilities = np.asarray(text_probabilities, dtype=np.float64).copy()
    if len(probabilities) != len(evidence):
        raise ValueError("text probabilities and retrieval evidence must align")
    if mode not in {
        "text",
        "hard",
        "soft",
        "hybrid",
        "tier_a",
        "tier_a_negative",
        "tier_a_positive",
    }:
        raise ValueError(f"unknown retrieval mode: {mode}")
    if mode == "text":
        return probabilities

    eps = 1e-5
    for i, item in enumerate(evidence):
        if mode.startswith("tier_a"):
            tier_a_label = item.get("tier_a_label")
            allowed = tier_a_label is not None and (
                mode == "tier_a"
                or (mode == "tier_a_negative" and int(tier_a_label) == 0)
                or (mode == "tier_a_positive" and int(tier_a_label) == 1)
            )
            if allowed:
                probabilities[i] = 1.0 - eps if int(tier_a_label) else eps
            continue
        hard_label = item.get("hard_label")
        if mode in {"hard", "hybrid"} and hard_label is not None:
            probabilities[i] = 1.0 - eps if int(hard_label) else eps
            continue
        if mode not in {"soft", "hybrid"}:
            continue
        support = int(item.get("support", 0))
        purity = float(item.get("purity", 0.0))
        if not support or purity < 0.8:
            continue

        text_prob = float(np.clip(probabilities[i], eps, 1 - eps))
        text_logit = np.log(text_prob / (1 - text_prob))
        retrieval_prob = float(np.clip(item.get("posterior", 0.5), eps, 1 - eps))
        retrieval_logit = np.log(retrieval_prob / (1 - retrieval_prob))
        reliability = purity * min(1.0, np.log1p(support) / np.log(4.0))
        combined = text_logit + float(weight) * reliability * retrieval_logit
        probabilities[i] = 1.0 / (1.0 + np.exp(-np.clip(combined, -30.0, 30.0)))
    return probabilities


def apply_empirical_absence_veto(
    probabilities: Sequence[float],
    categories: Sequence[object],
    flag_rows: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen flammable absence veto identically in train/inference."""

    output = np.asarray(probabilities, dtype=np.float64).copy()
    if not (len(output) == len(categories) == len(flag_rows)):
        raise ValueError("veto inputs must align")
    mask = np.asarray(
        [
            str(category) == FLAMMABLE_CATEGORY
            and bool(flags.get("empirical_absence_veto"))
            for category, flags in zip(categories, flag_rows)
        ],
        dtype=bool,
    )
    output[mask] = 1e-5
    return output, mask


def predict_by_category(artifact: Mapping[str, object], dataframe) -> tuple[np.ndarray, np.ndarray]:
    """Apply category models and their OOF thresholds while preserving row order."""

    probabilities = np.zeros(len(dataframe), dtype=np.float64)
    predictions = np.zeros(len(dataframe), dtype=np.int8)
    models = artifact["text_models"]

    for category, model_info in models.items():
        mask = dataframe["category"].astype(str).eq(category).to_numpy()
        positions = np.flatnonzero(mask)
        if not len(positions):
            continue
        subset = dataframe.iloc[positions]
        probs = predict_text_probabilities(
            model_info,
            subset["name"].tolist(),
            subset["description"].tolist(),
            subset["category"].tolist(),
        )
        probabilities[positions] = probs
        predictions[positions] = (probs >= float(model_info["threshold"])).astype(np.int8)

    return probabilities, predictions
