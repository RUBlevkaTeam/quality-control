"""Fast CPU-friendly TF-IDF baseline used for local experiments."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,19}(?:\s[^>]{0,300})?/?>")
_WS_RE = re.compile(r"\s+")


def clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).map(html.unescape).map(html.unescape)
    cleaned = cleaned.str.replace(_TAG_RE, " ", regex=True)
    return cleaned.str.replace(_WS_RE, " ", regex=True).str.strip()


def build_model_text(df: pd.DataFrame) -> pd.Series:
    required = {"name", "description", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing text columns: {sorted(missing)}")
    name = clean_text(df["name"])
    description = clean_text(df["description"])
    category = clean_text(df["category"])
    return name + "\n" + name + "\n" + category + "\n" + description


@dataclass(frozen=True)
class TextBaselineConfig:
    word_max_features: int = 40_000
    char_max_features: int = 60_000
    min_df: int = 2
    c: float = 1.0
    class_weight: str | None = "balanced"
    random_state: int = 42


class TextQualityPredictor:
    """Shared TF-IDF features with one binary classifier per category."""

    def __init__(self, config: TextBaselineConfig | None = None):
        self.config = config or TextBaselineConfig()
        self.vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=self.config.min_df,
                        max_df=0.995,
                        max_features=self.config.word_max_features,
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=self.config.min_df,
                        max_df=0.995,
                        max_features=self.config.char_max_features,
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        self.category_models: dict[str, LogisticRegression] = {}
        self.thresholds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "TextQualityPredictor":
        if "label" not in df.columns:
            raise ValueError("training dataframe must contain label")
        texts = build_model_text(df)
        features = self.vectorizer.fit_transform(texts)
        self.category_models = {}
        for category in sorted(df["category"].astype(str).unique()):
            mask = df["category"].astype(str).eq(category).to_numpy()
            labels = df.loc[mask, "label"].to_numpy()
            if len(np.unique(labels)) != 2:
                raise ValueError(f"category {category!r} does not contain both labels")
            model = LogisticRegression(
                C=self.config.c,
                class_weight=self.config.class_weight,
                max_iter=1_000,
                solver="liblinear",
                random_state=self.config.random_state,
            )
            model.fit(features[mask], labels)
            self.category_models[category] = model
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.category_models:
            raise RuntimeError("predictor is not fitted")
        features = self.vectorizer.transform(build_model_text(df))
        categories = df["category"].astype(str).to_numpy()
        probabilities = np.zeros(len(df), dtype=np.float64)
        for category in np.unique(categories):
            if category not in self.category_models:
                raise ValueError(f"unknown category: {category!r}")
            mask = categories == category
            probabilities[mask] = self.category_models[category].predict_proba(
                features[mask]
            )[:, 1]
        return probabilities

