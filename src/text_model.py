"""Категорийный sparse-классификатор текста и rule-признаков.

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

from src.rule_features import (
    FLAMMABLE_CATEGORY,
    build_model_texts as build_normalized_model_texts,
    rule_feature_matrix,
)
from src.retrieval_memory import (
    RetrievalMemory,
    absence_veto_mask,
    strict_positive_mask,
)
from src.utils_data_prep import _clean
from src.utils_logreg import ProductQualityPredictor, _norm_category

TEXT_MODEL_FORMAT_VERSION = 3

# БАД оставляем на проверенной word-only голове. Для редкого класса
# «Легковоспламеняющиеся» добавляем устойчивые к опечаткам и вариантам написания
# char-каналы; на одинаковых family-folds они дали стабильный прирост.
_BASIC_RULE_WEIGHT = 2.0
_RICH_RULE_WEIGHT = 3.0
_BASIC_MODE = "word_rules"
_RICH_MODE = "word_char_title_rules"

# Название повторено трижды: для TF-IDF это вес признаков заголовка.
# Замер на data.csv: +0.010 среднего F1, почти весь прирост на редкой
# категории (0.797 -> 0.818).
_TITLE_REPEATS = 3

# e5-канал: второй, ортогональный голос. Соло слабее TF-IDF, но ошибается в
# других местах. Family-фолды, 4 сида: fire 0.7986 -> 0.8046 (+0.0061),
# у БАД канал ничего не дал (-0.0004), поэтому включён только для fire.
# ВЫКЛЮЧЕН. Прирост OOF +0.011 оказался артефактом подбора порога, а не
# лучшего ранжирования: на OOF канал переворачивает 5 вердиктов из 3910
# (~1 товар на public), при этом весит 1.1 ГБ архива. Сабмит с ним:
# 0.77248 -> 0.76204. Код оставлен как задокументированный отрицательный
# результат; чтобы включить - вернуть FLAMMABLE_CATEGORY в _E5_CATEGORIES.
_E5_WEIGHT = 0.5
_E5_CATEGORIES = ()

# Полный двусторонний memory-lookup (текст/SHA/dHash). Выключен для
# изоляционного замера: сабмит со всеми слоями дал 0.77196 против 0.77329
# у базы, а это единственный компонент без OOF-замера (групповые фолды его
# не видят по построению). Негативная desc-память НЕ под этим флагом - её
# вклад замерен отдельно (fire +0.006..0.012, исправлено/испорчено 10/0).
_USE_FULL_MEMORY = False


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


# TF-IDF + rule-признаки одной sparse-матрицей; используется и в train, и в predict.
def _rule_matrix(df: pd.DataFrame, weight: float):
    from scipy import sparse

    names = df["name"].tolist() if "name" in df.columns else [""] * len(df)
    descs = df["description"].tolist() if "description" in df.columns else [""] * len(df)
    cats = df["category"].tolist() if "category" in df.columns else [""] * len(df)
    rules = rule_feature_matrix(names, descs, cats) * np.float32(weight)
    return sparse.csr_matrix(rules)


def _build_basic_features(vectorizer, df: pd.DataFrame, fit: bool):
    from scipy import sparse

    texts = build_model_texts(df).values
    tfidf = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    return sparse.hstack(
        (tfidf, _rule_matrix(df, _BASIC_RULE_WEIGHT)), format="csr"
    )


def _build_rich_features(vectorizers: dict, df: pd.DataFrame, fit: bool):
    """Word + body char + title char channels for the rare fire category."""

    from scipy import sparse

    names = df["name"].tolist() if "name" in df.columns else [""] * len(df)
    descs = df["description"].tolist() if "description" in df.columns else [""] * len(df)
    texts = build_normalized_model_texts(names, descs)
    title_texts = build_normalized_model_texts(names, [""] * len(df))

    transform = "fit_transform" if fit else "transform"
    word = getattr(vectorizers["word"], transform)(texts)
    char = getattr(vectorizers["char"], transform)(texts)
    title = getattr(vectorizers["title"], transform)(title_texts)
    return sparse.hstack(
        (word, char, title, _rule_matrix(df, _RICH_RULE_WEIGHT)),
        format="csr",
        dtype=np.float32,
    )


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
        families: pd.Series | None = None,
        c_grid: Sequence[float] = (0.3, 1.0, 3.0),
        n_splits: int = 5,
        random_state: int = 42,
        extra_threshold_seeds: Sequence[int] = (3030, 3031, 3032),
        e5_model_path: str | None = None,
    ) -> Dict[str, dict]:
        """families: Series id -> family. С ней фолды режутся по семействам
        почти-дублей (53% товаров в семействах!) - без этого OOF завышен:
        0.871 против 0.687 на лидерборде."""
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

        self.category_models = {}
        report: Dict[str, dict] = {}
        fam_by_id = None if families is None else dict(zip(families["id"], families["family"]))

        for category in train_df["category"].dropna().unique():
            subset = train_df[train_df["category"] == category]
            # дедуп тот же, что у эмбеддингового классификатора: точные копии
            # с одинаковой меткой рвутся сплитом и завышают OOF
            subset = ProductQualityPredictor._dedup(subset).reset_index(drop=True)

            y = np.asarray(subset["label"].values, dtype=np.int64)
            n_pos = int(y.sum())
            if n_pos == 0 or n_pos == len(y):
                _log(f"{category}: один класс ({n_pos}/{len(y)}), категория пропущена")
                continue

            splits = max(2, min(n_splits, n_pos))
            if fam_by_id is not None:
                groups = np.asarray([fam_by_id.get(i, -1) for i in subset["id"]])
                cv = StratifiedGroupKFold(splits, shuffle=True, random_state=random_state)
                folds = list(cv.split(subset, y, groups))
            else:
                cv = StratifiedKFold(splits, shuffle=True, random_state=random_state)
                folds = list(cv.split(subset, y))

            # e5-эмбеддинги нужны только там, где канал что-то даёт (fire);
            # считаются один раз на категорию и переиспользуются по всем фолдам
            e5_matrix = None
            if self._uses_e5(category) and e5_model_path is not None:
                try:
                    from src.e5_channel import build_e5_texts, encode

                    e5_matrix = encode(build_e5_texts(subset), e5_model_path)
                    _log(f"{category}: e5-эмбеддинги {e5_matrix.shape}")
                except Exception as exc:
                    _log(f"{category}: e5-канал недоступен ({type(exc).__name__}: {exc})")

            def oof_for(seed: int, C: float):
                """OOF-вероятности для сида: базовые признаки, при наличии
                e5 - логит-смесь. Порог обязан считаться на том же, что
                увидит инференс, иначе он не соответствует предсказаниям."""
                if fam_by_id is not None:
                    cv_seed = StratifiedGroupKFold(splits, shuffle=True, random_state=seed)
                    split = list(cv_seed.split(subset, y, groups))
                else:
                    cv_seed = StratifiedKFold(splits, shuffle=True, random_state=seed)
                    split = list(cv_seed.split(subset, y))

                base = np.zeros(len(y), dtype=np.float64)
                e5_oof = np.zeros(len(y), dtype=np.float64) if e5_matrix is not None else None
                for tr, va in split:
                    features = self._make_feature_bundle(category)
                    x_tr = self._build_category_features(
                        category, features, subset.iloc[tr], fit=True
                    )
                    x_va = self._build_category_features(
                        category, features, subset.iloc[va], fit=False
                    )
                    clf = self._make_classifier(category, C)
                    clf.fit(x_tr, y[tr])
                    base[va] = clf.predict_proba(x_va)[:, 1]

                    if e5_matrix is not None:
                        head = self._make_e5_head()
                        head.fit(e5_matrix[tr], y[tr])
                        e5_oof[va] = head.predict_proba(e5_matrix[va])[:, 1]

                if e5_oof is None:
                    return base
                from src.e5_channel import blend_logits

                return blend_logits(base, e5_oof, _E5_WEIGHT)

            best = None
            for C in c_grid:
                f1, threshold = exact_best_threshold(oof_for(random_state, C), y)
                if best is None or f1 > best["oof_f1"]:
                    best = {"oof_f1": f1, "threshold": threshold, "C": C}

            if best is None or best["oof_f1"] <= 0.0:
                raise RuntimeError(f"{category}: ни одна конфигурация не дала F1 > 0")

            # Multi-seed порог: разбиение с одним seed даёт шумный порог на
            # 143 позитивах (у нас же измерено: std порога ~0.05-0.1). Медиана
            # по 4 независимым разбиениям устойчивее - важно перед private.
            if extra_threshold_seeds and fam_by_id is not None:
                thresholds = [best["threshold"]]
                f1s = [best["oof_f1"]]
                for seed in extra_threshold_seeds:
                    f1_2, thr_2 = exact_best_threshold(oof_for(seed, best["C"]), y)
                    thresholds.append(thr_2)
                    f1s.append(f1_2)
                best["threshold"] = float(np.median(thresholds))
                best["oof_f1"] = float(np.mean(f1s))
                best["threshold_seeds"] = [random_state, *extra_threshold_seeds]
                _log(
                    f"{category}: multi-seed пороги {np.round(thresholds, 4).tolist()} "
                    f"-> медиана {best['threshold']:.4f}; F1 по сидам "
                    f"{np.round(f1s, 4).tolist()}"
                )

            features = self._make_feature_bundle(category)
            matrix = self._build_category_features(category, features, subset, fit=True)
            classifier = self._make_classifier(category, best["C"])
            classifier.fit(matrix, y)

            # retrieval-память: групповой OOF её вклад не видит по построению
            # (дубли всегда в одном фолде); перенос метки только при единогласии,
            # на train точность 0.9992. Хранится опциональным полем.
            memory = RetrievalMemory().fit(subset)

            e5_head = None
            if e5_matrix is not None:
                e5_head = self._make_e5_head()
                e5_head.fit(e5_matrix, y)

            self.category_models[category] = {
                "feature_mode": self._feature_mode(category),
                "features": features,
                "memory": memory,
                "e5_head": e5_head,
                "classifier": classifier,
                "threshold": best["threshold"],
                "C": best["C"],
                "oof_f1": best["oof_f1"],
                "n_train": int(len(y)),
                "n_pos": n_pos,
            }
            report[category] = {
                k: v for k, v in self.category_models[category].items()
                if k not in ("features", "classifier", "memory", "e5_head")
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
    def _make_basic_vectorizer():
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

    @staticmethod
    def _make_rich_vectorizers() -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer

        return {
            "word": TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.999,
                max_features=100_000,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
                dtype=np.float32,
            ),
            "char": TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 6),
                min_df=2,
                max_df=0.999,
                max_features=160_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "title": TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 6),
                min_df=2,
                max_df=0.999,
                max_features=70_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
        }

    @staticmethod
    def _feature_mode(category: object) -> str:
        return _RICH_MODE if str(category) == FLAMMABLE_CATEGORY else _BASIC_MODE

    @classmethod
    def _make_feature_bundle(cls, category: object) -> dict:
        if cls._feature_mode(category) == _RICH_MODE:
            return cls._make_rich_vectorizers()
        return {"word": cls._make_basic_vectorizer()}

    @classmethod
    def _build_category_features(
        cls, category_or_mode: object, features: dict, df: pd.DataFrame, fit: bool
    ):
        value = str(category_or_mode)
        mode = value if value in {_BASIC_MODE, _RICH_MODE} else cls._feature_mode(value)
        if mode == _RICH_MODE:
            return _build_rich_features(features, df, fit)
        return _build_basic_features(features["word"], df, fit)

    @staticmethod
    def _make_classifier(category: object, C: float):
        from sklearn.linear_model import LogisticRegression

        if str(category) != FLAMMABLE_CATEGORY:
            # БАД: логрег. Пробовали калиброванный LinearSVC - на family-OOF
            # он выигрывал стабильно (0.9370 -> 0.9407 на всех 4 сидах), но
            # менял ~17 вердиктов на public, и сабмит просел 0.77248 -> 0.76204.
            # Разделить "объективно хуже" и "не повезло с 17 товарами" одним
            # публичным замером нельзя, поэтому остаёмся на проверенном.
            return LogisticRegression(
                C=C, max_iter=2000, class_weight="balanced",
            )

        # fire: фиксированный вес позитива 40 вместо balanced (~13.7). Меняет
        # ранжирование редких позитивов, порог после этого отделяет уверенные
        # fire-товары чище. Family-folds по 4 сидам: 0.79267 -> 0.79757.
        return LogisticRegression(
            C=C,
            max_iter=2000,
            solver="liblinear",
            random_state=2026,
            class_weight={0: 1, 1: 40},
        )

    @staticmethod
    def _uses_e5(category: object) -> bool:
        return str(category) in _E5_CATEGORIES

    # эмбеддинги считаются один раз на подвыборку категории
    @staticmethod
    def _e5_head_probs(head, sub: pd.DataFrame, e5_model_path):
        from src.e5_channel import build_e5_texts, encode

        if e5_model_path is None:
            raise ValueError("путь к e5-модели не задан")
        matrix = encode(build_e5_texts(sub), e5_model_path)
        return head.predict_proba(matrix)[:, 1]

    @staticmethod
    def _make_e5_head():
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=3.0, max_iter=3000, class_weight="balanced")

    # ------------------------------------------------------------------
    # инференс
    # ------------------------------------------------------------------

    # Никогда не бросает исключение: незнакомая категория или пустой вход
    # деградируют до prob=0/pred=0, потому что упавший прогон стоит дороже.
    def predict(
        self, df: pd.DataFrame, e5_model_path: str | None = None
    ) -> Tuple[List[float], List[int]]:
        n = len(df)
        probs = np.zeros(n, dtype=np.float64)
        preds = np.zeros(n, dtype=np.int64)
        if n == 0 or not self.category_models:
            return probs.tolist(), preds.tolist()

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
            sub = df.iloc[index]
            matrix = self._build_category_features(
                info["feature_mode"], info["features"], sub, fit=False
            )
            block = info["classifier"].predict_proba(matrix)[:, 1]

            # e5-канал: второй голос, ортогональный TF-IDF. Порог в артефакте
            # подобран на СМЕШАННЫХ вероятностях, поэтому при сбое канала
            # предсказания поедут по другой шкале - логируем это явно.
            e5_head = info.get("e5_head")
            if e5_head is not None:
                try:
                    from src.e5_channel import blend_logits, build_e5_texts, encode

                    e5_probs = self._e5_head_probs(e5_head, sub, e5_model_path)
                    block = blend_logits(block, e5_probs, _E5_WEIGHT)
                except Exception as exc:
                    _log(
                        f"e5-канал не отработал ({type(exc).__name__}: {exc}); "
                        "остаёмся на текстовой модели, порог рассчитан на смеси"
                    )

            # retrieval-память: точное совпадение с train перекрывает модель
            memory = info.get("memory")
            if memory is not None and _USE_FULL_MEMORY:
                try:
                    got, _src = memory.lookup(sub)
                    hits = sum(1 for g in got if g is not None)
                    for j, g in enumerate(got):
                        if g is not None:
                            block[j] = 1.0 - 1e-5 if int(g) == 1 else 1e-5
                    if hits:
                        _log(f"память перенесла вердикт для {hits} товаров")
                except Exception as exc:
                    _log(f"память недоступна ({type(exc).__name__}: {exc}), продолжаю без неё")

            # негативный desc-lookup: только fire, только в сторону pred=0
            if memory is not None and key == _norm_category(FLAMMABLE_CATEGORY):
                try:
                    neg = memory.negative_desc_mask(sub)
                    if neg.any():
                        block[neg] = np.minimum(block[neg], 1e-5)
                        _log(f"neg-desc память обнулила {int(neg.sum())} товаров")
                except Exception as exc:
                    _log(f"neg-desc память недоступна ({type(exc).__name__}: {exc})")

            probs[index] = block
            preds[index] = (block >= float(info["threshold"])).astype(np.int64)

        # strict positive: высокоточное fire-семейство (37/37 на train) -> pred=1;
        # применяется ДО veto, чтобы veto оставался сильнее при пересечении
        try:
            pos = strict_positive_mask(df)
            if pos.any():
                probs[pos] = np.maximum(probs[pos], 1.0 - 1e-5)
                preds[pos] = 1
                _log(f"strict-positive применён к {int(pos.sum())} товарам")
        except Exception as exc:
            _log(f"strict-positive недоступен ({type(exc).__name__}: {exc})")

        # absence veto: «топливо не входит в комплект» у легковоспламеняющихся ->
        # принудительный pred=0. На train: 1208 срабатываний, 0 убитых позитивов.
        try:
            veto = absence_veto_mask(df)
            if veto.any():
                probs[veto] = 1e-5
                preds[veto] = 0
                _log(f"veto применён к {int(veto.sum())} товарам")
        except Exception as exc:
            _log(f"veto недоступен ({type(exc).__name__}: {exc}), продолжаю без него")

        if unknown:
            _log(f"{unknown} товаров с неизвестной категорией -> pred=0")
        return probs.tolist(), preds.tolist()

    def summary(self) -> str:
        if not self.category_models:
            return "текстовая модель пуста"
        parts = []
        for name, info in self.category_models.items():
            vocab = sum(len(vec.vocabulary_) for vec in info["features"].values())
            parts.append(
                f"{name}: {info['feature_mode']}, порог {info['threshold']:.3f}, "
                f"словарь {vocab}, OOF {info.get('oof_f1', 0):.3f}"
            )
        return "; ".join(parts)

    def thresholds(self) -> Dict[str, float]:
        """Порог по нормализованному ключу категории - для гейта OCR."""

        return {
            _norm_category(name): float(info["threshold"])
            for name, info in self.category_models.items()
        }

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
            for key in ("feature_mode", "features", "classifier", "threshold"):
                if key not in info:
                    raise ValueError(f"{name!r}: в артефакте нет поля {key!r}")
            mode = info["feature_mode"]
            required_features = (
                {"word", "char", "title"} if mode == _RICH_MODE else {"word"}
            )
            if mode not in {_BASIC_MODE, _RICH_MODE}:
                raise ValueError(f"{name!r}: неизвестный feature_mode {mode!r}")
            missing = sorted(required_features - set(info["features"]))
            if missing:
                raise ValueError(f"{name!r}: не хватает векторизаторов {missing}")
            threshold = float(info["threshold"])
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name!r}: некорректный порог {threshold}")


# Обучение из командной строки:
#   venv312/bin/python -m src.text_model data.csv text_model.joblib
def _main(argv: Sequence[str]) -> None:
    if len(argv) not in (2, 3, 4):
        raise SystemExit(
            "использование: python -m src.text_model <data.csv> <выходной.joblib> "
            "[families.csv] [label_corrections.csv]"
        )
    # при запуске через -m этот модуль называется __main__, и класс запиклился
    # бы как __main__.TextQualityModel - такой артефакт не загрузится из run.py.
    # Импортируем класс под каноническим именем модуля.
    from src.text_model import TextQualityModel as CanonicalTextQualityModel

    data_path, out_path = argv[0], argv[1]
    families = pd.read_csv(argv[2]) if len(argv) >= 3 else None
    # prepare_dataframe, а не голый read_csv: без image_paths память
    # не построит SHA/dHash-ключи и будет работать только по тексту
    from pathlib import Path

    from src.utils_data_prep import prepare_dataframe

    data_path = Path(data_path)
    df = prepare_dataframe(data_path, data_path.parent / "images")

    # ручной арбитраж конфликтных семейств (одинаковый контент, разные метки):
    # правки применяются к обучению и к памяти, файл - артефакт решения
    if len(argv) == 4:
        corrections = pd.read_csv(argv[3])
        fix = dict(zip(corrections["id"], corrections["corrected_label"]))
        mask = df["id"].isin(fix)
        df.loc[mask, "label"] = df.loc[mask, "id"].map(fix)
        _log(f"применено {int(mask.sum())} правок разметки из {argv[3]}")
    e5_dir = Path(__file__).resolve().parent.parent / "models" / "multilingual-e5-base"
    model = CanonicalTextQualityModel()
    model.train(df, families=families,
                e5_model_path=str(e5_dir) if e5_dir.exists() else None)
    model.save(out_path)
    _log(f"артефакт сохранён: {out_path}")


if __name__ == "__main__":
    _main(sys.argv[1:])
