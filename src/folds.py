"""Deterministic group-aware cross-validation folds."""

from __future__ import annotations

import hashlib
import html
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,19}(?:\s[^>]{0,300})?/?>")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zа-я]+")


def normalize_for_grouping(series: pd.Series) -> pd.Series:
    """Normalize product text aggressively enough to group exact-like duplicates."""
    normalized = series.fillna("").astype(str).map(html.unescape).map(html.unescape)
    normalized = normalized.str.replace(_TAG_RE, " ", regex=True).str.lower()
    normalized = normalized.str.replace("ё", "е", regex=False)
    return normalized.str.replace(_NON_ALNUM_RE, "", regex=True)


def make_group_keys(df: pd.DataFrame) -> pd.Series:
    required = {"name", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns for grouping: {sorted(missing)}")
    text = normalize_for_grouping(df["name"]) + "\n" + normalize_for_grouping(
        df["description"]
    )
    return text.map(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest())


def build_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return id, group_key and fold while keeping duplicate groups together."""
    required = {"id", "category", "label", "name", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns for folds: {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError("id values must be unique")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    groups = make_group_keys(df)
    strata = df["category"].astype(str) + "__" + df["label"].astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = np.full(len(df), -1, dtype=np.int8)
    for fold, (_, valid_indices) in enumerate(
        splitter.split(np.zeros(len(df)), strata, groups)
    ):
        folds[valid_indices] = fold
    if (folds < 0).any():
        raise RuntimeError("some rows were not assigned to a fold")

    result = pd.DataFrame(
        {"id": df["id"].to_numpy(), "group_key": groups.to_numpy(), "fold": folds}
    )
    if result.groupby("group_key")["fold"].nunique().max() != 1:
        raise RuntimeError("group leakage detected between folds")
    return result

