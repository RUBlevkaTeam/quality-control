"""Per-category logistic regression used locally and in the submission."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

_NPZ_FORMAT_VERSION = 1


def _category_setting(value, category: str):
    if isinstance(value, Mapping):
        if category not in value:
            raise ValueError(f"missing setting for category: {category!r}")
        return value[category]
    return value


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    absolute = np.abs(logits)
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-absolute)),
        np.exp(-absolute) / (1.0 + np.exp(-absolute)),
    )


class ProductQualityPredictor:
    def __init__(self) -> None:
        self.category_models: dict[str, dict[str, object]] = {}

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        categories: np.ndarray,
        *,
        c: float | Mapping[str, float] = 1.0,
        normalize: bool | Mapping[str, bool] = False,
        random_state: int = 42,
    ) -> "ProductQualityPredictor":
        from sklearn.linear_model import LogisticRegression

        embeddings = np.asarray(embeddings)
        categories = np.asarray(categories).astype(str)
        labels = np.asarray(labels)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if not (len(embeddings) == len(labels) == len(categories)):
            raise ValueError("embeddings, labels and categories must have equal lengths")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain NaN or infinite values")

        self.category_models = {}
        for category_value in sorted(np.unique(categories)):
            category = str(category_value)
            mask = categories == category
            category_labels = labels[mask]
            if len(np.unique(category_labels)) != 2:
                raise ValueError(f"category {category!r} must contain both classes")
            category_c = float(_category_setting(c, category))
            category_normalize = bool(_category_setting(normalize, category))
            if not np.isfinite(category_c) or category_c <= 0:
                raise ValueError(f"C must be positive for category {category!r}")

            category_embeddings = embeddings[mask]
            if category_normalize:
                category_embeddings = _normalize_rows(category_embeddings)
            model = LogisticRegression(
                C=category_c,
                class_weight="balanced",
                max_iter=1_000,
                solver="liblinear",
                random_state=random_state,
            )
            model.fit(category_embeddings, category_labels)
            self.category_models[category] = {
                "model": model,
                "threshold": 0.5,
                "c": category_c,
                "normalize": category_normalize,
            }

        return self

    def set_thresholds(self, thresholds: Mapping[str, float]) -> None:
        for category, threshold in thresholds.items():
            if str(category) not in self.category_models:
                raise ValueError(f"unknown category: {category!r}")
            value = float(threshold)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid threshold for category {category!r}: {value}")
            self.category_models[str(category)]["threshold"] = value

    @staticmethod
    def _linear_parameters(model_info: Mapping[str, object]) -> tuple[np.ndarray, float]:
        model = model_info.get("model")
        if model is not None:
            return (
                np.asarray(model.coef_[0], dtype=np.float64),
                float(model.intercept_[0]),
            )
        return (
            np.asarray(model_info["coef"], dtype=np.float64),
            float(model_info["intercept"]),
        )

    def predict(self, embeddings, categories) -> tuple[np.ndarray, np.ndarray]:
        embeddings = np.asarray(embeddings)
        categories = np.asarray(categories).astype(str)
        if not self.category_models:
            raise RuntimeError("predictor is not fitted")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if len(embeddings) != len(categories):
            raise ValueError("embeddings and categories must have equal lengths")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain NaN or infinite values")
        unknown_categories = sorted(set(categories) - set(self.category_models))
        if unknown_categories:
            raise ValueError(f"unknown categories: {unknown_categories}")

        probabilities = np.zeros(len(embeddings), dtype=np.float64)
        predictions = np.zeros(len(embeddings), dtype=np.int8)
        for category, model_info in self.category_models.items():
            mask = categories == category
            if not mask.any():
                continue
            category_embeddings = embeddings[mask]
            if bool(model_info.get("normalize", False)):
                category_embeddings = _normalize_rows(category_embeddings)

            model = model_info.get("model")
            if model is not None:
                category_probabilities = model.predict_proba(category_embeddings)[:, 1]
            else:
                coefficients, intercept = self._linear_parameters(model_info)
                if category_embeddings.shape[1] != len(coefficients):
                    raise ValueError(
                        f"category {category!r} expects {len(coefficients)} features, "
                        f"got {category_embeddings.shape[1]}"
                    )
                category_probabilities = _sigmoid(
                    category_embeddings @ coefficients + intercept
                )

            probabilities[mask] = category_probabilities
            predictions[mask] = (
                category_probabilities >= float(model_info["threshold"])
            )

        return probabilities, predictions

    def summary(self) -> str:
        if not self.category_models:
            return "predictor is not fitted"
        parts = []
        for category, model_info in sorted(self.category_models.items()):
            coefficients, _ = self._linear_parameters(model_info)
            parts.append(
                f"{category}: threshold={float(model_info['threshold']):.2f}, "
                f"C={float(model_info.get('c', 1.0)):g}, "
                f"normalize={bool(model_info.get('normalize', False))}, "
                f"dim={len(coefficients)}"
            )
        return "; ".join(parts)

    def export_npz(self, filepath: str | Path) -> None:
        if not self.category_models:
            raise ValueError("cannot export an unfitted predictor")
        categories = sorted(self.category_models)
        parameters = [self._linear_parameters(self.category_models[c]) for c in categories]
        dimensions = {len(coefficients) for coefficients, _ in parameters}
        if len(dimensions) != 1:
            raise ValueError(f"category models have different dimensions: {sorted(dimensions)}")

        np.savez(
            filepath,
            format_version=np.array(_NPZ_FORMAT_VERSION, dtype=np.int16),
            categories=np.asarray(categories),
            coefficients=np.vstack([item[0] for item in parameters]),
            intercepts=np.asarray([item[1] for item in parameters], dtype=np.float64),
            thresholds=np.asarray(
                [self.category_models[c]["threshold"] for c in categories],
                dtype=np.float64,
            ),
            c_values=np.asarray(
                [self.category_models[c].get("c", 1.0) for c in categories],
                dtype=np.float64,
            ),
            normalize=np.asarray(
                [self.category_models[c].get("normalize", False) for c in categories],
                dtype=np.bool_,
            ),
        )

    @classmethod
    def from_npz(cls, filepath: str | Path) -> "ProductQualityPredictor":
        with np.load(filepath, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version != _NPZ_FORMAT_VERSION:
                raise ValueError(f"unsupported predictor format version: {version}")
            categories = data["categories"].astype(str)
            coefficients = np.asarray(data["coefficients"], dtype=np.float64)
            intercepts = np.asarray(data["intercepts"], dtype=np.float64)
            thresholds = np.asarray(data["thresholds"], dtype=np.float64)
            c_values = np.asarray(data["c_values"], dtype=np.float64)
            normalize = np.asarray(data["normalize"], dtype=np.bool_)

        count = len(categories)
        if coefficients.ndim != 2 or coefficients.shape[0] != count:
            raise ValueError("invalid coefficients shape in predictor artifact")
        if not all(len(values) == count for values in (intercepts, thresholds, c_values, normalize)):
            raise ValueError("predictor artifact arrays do not align")

        predictor = cls()
        for index, category in enumerate(categories):
            predictor.category_models[str(category)] = {
                "coef": coefficients[index],
                "intercept": float(intercepts[index]),
                "threshold": float(thresholds[index]),
                "c": float(c_values[index]),
                "normalize": bool(normalize[index]),
            }
        return predictor

    def save(self, filepath: str | Path) -> None:
        import joblib

        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "ProductQualityPredictor":
        path = Path(filepath)
        if path.suffix.lower() == ".npz":
            return cls.from_npz(path)
        import joblib

        return joblib.load(path)
