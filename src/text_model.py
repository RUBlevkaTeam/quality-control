"""Текстовый классификатор: TF-IDF по name+description и логрег на категорию.

Основной источник вердикта. По OOF-замерам на data.csv текст даёт средний
F1 ~0.86 против 0.505 у эмбеддингового пути на лидерборде, при этом не требует
ни GPU, ни /shared_models: инференс на 3800 товарах занимает секунды на CPU.

Обучение и инференс живут в одном модуле и используют один и тот же
препроцессинг (_clean из utils_data_prep + _build_model_text ниже), чтобы
подготовка текста при обучении и в контейнере не разъехались молча.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils_data_prep import _clean
from src.utils_logreg import ProductQualityPredictor, _norm_category

TEXT_MODEL_FORMAT_VERSION = 1

# Название повторено трижды: для TF-IDF это вес признаков заголовка.
# Замер на data.csv: +0.010 среднего F1, почти весь прирост на редкой
# категории (0.797 -> 0.818).
_TITLE_REPEATS = 3


def _log(message: str) -> None:
    print(f"[text_model] {message}", file=sys.stderr, flush=True)


# единственное место, где собирается вход векторизатора
def build_model_texts(df: pd.DataFrame) -> pd.Series:
    name = _clean(df["name"]) if "name" in df.columns else pd.Series([""] * len(df))
    desc = _clean(df["description"]) if "description" in df.columns else pd.Series([""] * len(df))
    name = name.reset_index(drop=True)
    desc = desc.reset_index(drop=True)
    title = ("Название: " + name + " ") * _TITLE_REPEATS
    return (title + "Описание: " + desc).str.strip()


# Точный поиск порога: F1 меняется только на наблюдаемых значениях
# вероятностей, поэтому сортировка + кумулятивные TP дают оптимум за
# O(n log n) без сетки. Возвращает (лучший F1, медиана плато порогов).
def exact_best_threshold(probs: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    positives = int(y.sum())
    if positives == 0 or len(y) == 0:
        return 0.0, 0.5
    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order]
    sorted_y = y[order]
    tp = np.cumsum(sorted_y)
    k = np.arange(1, len(y) + 1)
    f1 = 2.0 * tp / (k + positives)
    # резать можно только там, где вероятность реально меняется
    boundary = np.r_[sorted_probs[:-1] != sorted_probs[1:], True]
    f1 = np.where(boundary, f1, -1.0)
    best = float(f1.max())
    if best <= 0.0:
        return 0.0, 0.5
    tied = np.flatnonzero(f1 >= best - 1e-12)
    return best, float(np.median(sorted_probs[tied]))


class TextQualityModel:
    """{категория: {'vectorizer', 'classifier', 'threshold', ...}}"""

    def __init__(self) -> None:
        self.category_models: Dict[str, dict] = {}
        self.format_version = TEXT_MODEL_FORMAT_VERSION

    # ------------------------------------------------------------------
    # обучение (в контейнере не выполняется)
    # ------------------------------------------------------------------

    def train(
        self,
        train_df: pd.DataFrame,
        *,
        c_grid: Sequence[float] = (0.3, 1.0, 3.0),
        n_splits: int = 5,
        random_state: int = 42,
    ) -> Dict[str, dict]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline

        self.category_models = {}
        report: Dict[str, dict] = {}

        for category in train_df["category"].dropna().unique():
            subset = train_df[train_df["category"] == category]
            # дедуп тот же, что у эмбеддингового классификатора: точные копии
            # с одинаковой меткой рвутся сплитом и завышают OOF
            subset = ProductQualityPredictor._dedup(subset)

            texts = build_model_texts(subset).values
            y = np.asarray(subset["label"].values, dtype=np.int64)
            n_pos = int(y.sum())
            if n_pos == 0 or n_pos == len(y):
                _log(f"{category}: один класс ({n_pos}/{len(y)}), категория пропущена")
                continue

            splits = max(2, min(n_splits, n_pos))
            best = None
            for C in c_grid:
                pipe = make_pipeline(
                    self._make_vectorizer(),
                    LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
                )
                cv = StratifiedKFold(splits, shuffle=True, random_state=random_state)
                oof = cross_val_predict(pipe, texts, y, cv=cv, method="predict_proba")[:, 1]
                f1, threshold = exact_best_threshold(oof, y)
                if best is None or f1 > best["oof_f1"]:
                    best = {"oof_f1": f1, "threshold": threshold, "C": C}

            if best is None or best["oof_f1"] <= 0.0:
                raise RuntimeError(f"{category}: ни одна конфигурация не дала F1 > 0")

            vectorizer = self._make_vectorizer()
            matrix = vectorizer.fit_transform(texts)
            classifier = LogisticRegression(
                C=best["C"], max_iter=2000, class_weight="balanced"
            )
            classifier.fit(matrix, y)

            self.category_models[category] = {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "threshold": best["threshold"],
                "C": best["C"],
                "oof_f1": best["oof_f1"],
                "n_train": int(len(y)),
                "n_pos": n_pos,
            }
            report[category] = {
                k: v for k, v in self.category_models[category].items()
                if k not in ("vectorizer", "classifier")
            }
            _log(
                f"{category}: OOF F1={best['oof_f1']:.4f}, порог={best['threshold']:.4f}, "
                f"C={best['C']:g}, n={len(y)} (позитивов {n_pos})"
            )

        if report:
            expected = train_df["category"].dropna().nunique()
            mean_f1 = sum(r["oof_f1"] for r in report.values()) / expected
            _log(f"ИТОГО средний OOF F1: {mean_f1:.4f}")
        return report

    @staticmethod
    def _make_vectorizer():
        from sklearn.feature_extraction.text import TfidfVectorizer

        # word 1-2 граммы: по замерам не уступают char_wb 3-6 (0.8610 против
        # 0.8611), но обучаются и применяются в ~7 раз быстрее
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # инференс
    # ------------------------------------------------------------------

    # Никогда не бросает исключение: незнакомая категория или пустой вход
    # деградируют до prob=0/pred=0, потому что упавший прогон стоит дороже.
    def predict(self, df: pd.DataFrame) -> Tuple[List[float], List[int]]:
        n = len(df)
        probs = np.zeros(n, dtype=np.float64)
        preds = np.zeros(n, dtype=np.int64)
        if n == 0 or not self.category_models:
            return probs.tolist(), preds.tolist()

        texts = build_model_texts(df).values
        categories = df["category"].tolist() if "category" in df.columns else [""] * n
        keys = [_norm_category(c) for c in categories]
        heads = {_norm_category(name): info for name, info in self.category_models.items()}

        groups: Dict[str, List[int]] = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)

        unknown = 0
        for key, positions in groups.items():
            info = heads.get(key)
            if info is None:
                unknown += len(positions)
                continue
            index = np.asarray(positions, dtype=np.int64)
            matrix = info["vectorizer"].transform(texts[index])
            block = info["classifier"].predict_proba(matrix)[:, 1]
            probs[index] = block
            preds[index] = (block >= float(info["threshold"])).astype(np.int64)

        if unknown:
            _log(f"{unknown} товаров с неизвестной категорией -> pred=0")
        return probs.tolist(), preds.tolist()

    def summary(self) -> str:
        if not self.category_models:
            return "текстовая модель пуста"
        parts = [
            f"{name}: порог {info['threshold']:.3f}, "
            f"словарь {len(info['vectorizer'].vocabulary_)}, OOF {info.get('oof_f1', 0):.3f}"
            for name, info in self.category_models.items()
        ]
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # сериализация
    # ------------------------------------------------------------------

    # Артефакт хранится через joblib. Риск несовместимости версий sklearn
    # здесь принят осознанно: прогон 0.505 подтвердил, что контейнер
    # распаковывает pickle от sklearn 1.9 (артефакт организаторов) и
    # выполняет predict_proba. Валидация ниже ловит битый артефакт при
    # загрузке, а run.py при любом сбое деградирует на эмбеддинговый путь.
    def save(self, filepath) -> None:
        import joblib

        if not self.category_models:
            raise ValueError("нечего сохранять: модель не обучена")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath) -> "TextQualityModel":
        import joblib

        model = joblib.load(filepath)
        cls.validate(model)
        return model

    @staticmethod
    def validate(model: "TextQualityModel") -> None:
        if getattr(model, "format_version", None) != TEXT_MODEL_FORMAT_VERSION:
            raise ValueError("неподдерживаемая версия текстовой модели")
        if not getattr(model, "category_models", None):
            raise ValueError("в артефакте нет ни одной категории")
        for name, info in model.category_models.items():
            for key in ("vectorizer", "classifier", "threshold"):
                if key not in info:
                    raise ValueError(f"{name!r}: в артефакте нет поля {key!r}")
            threshold = float(info["threshold"])
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name!r}: некорректный порог {threshold}")


# Обучение из командной строки:
#   venv312/bin/python -m src.text_model data.csv text_model.joblib
def _main(argv: Sequence[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("использование: python -m src.text_model <data.csv> <выходной.joblib>")
    # при запуске через -m этот модуль называется __main__, и класс запиклился
    # бы как __main__.TextQualityModel - такой артефакт не загрузится из run.py.
    # Импортируем класс под каноническим именем модуля.
    from src.text_model import TextQualityModel as CanonicalTextQualityModel

    data_path, out_path = argv
    df = pd.read_csv(data_path)
    junk = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if junk:
        df = df.drop(columns=junk)
    model = CanonicalTextQualityModel()
    model.train(df)
    model.save(out_path)
    _log(f"артефакт сохранён: {out_path}")


if __name__ == "__main__":
    _main(sys.argv[1:])
