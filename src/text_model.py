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

from src.retrieval_memory import RetrievalMemory, absence_veto_mask
from src.rule_features import FLAMMABLE_CATEGORY, rule_feature_matrix
from src.utils_data_prep import _clean
from src.utils_logreg import ProductQualityPredictor, _norm_category

TEXT_MODEL_FORMAT_VERSION = 3

# вес rule-признаков относительно TF-IDF, как у ultra (проверено её 0.8)
_RULE_WEIGHT = 2.0

# C=3.0 побеждал во всех замерах на групповой CV (обе категории, оба векторизатора)
_C = 3.0

# два взгляда на текст: word ловит термины, char_wb - морфологию внутри
# незнакомых слов (обобщение на новые бренды). Среднее их вероятностей дало
# +0.005 среднего F1 на групповой CV (редкий класс 0.764 -> 0.772)
_VECTORIZERS = ("word", "char")

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


# TF-IDF + rule-признаки одной sparse-матрицей; используется и в train, и в predict
def _build_features(vectorizer, texts, df: pd.DataFrame, fit: bool):
    from scipy import sparse

    tfidf = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    names = df["name"].tolist() if "name" in df.columns else [""] * len(df)
    descs = df["description"].tolist() if "description" in df.columns else [""] * len(df)
    cats = df["category"].tolist() if "category" in df.columns else [""] * len(df)
    rules = rule_feature_matrix(names, descs, cats) * np.float32(_RULE_WEIGHT)
    return sparse.hstack((tfidf, sparse.csr_matrix(rules)), format="csr")


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
        n_splits: int = 5,
        random_state: int = 42,
    ) -> Dict[str, dict]:
        """families: Series id -> family. С ней фолды режутся по семействам
        почти-дублей (53% товаров в семействах!) - без этого OOF завышен:
        0.871 против 0.687 на лидерборде."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

        self.category_models = {}
        report: Dict[str, dict] = {}
        fam_by_id = None if families is None else dict(zip(families["id"], families["family"]))

        for category in train_df["category"].dropna().unique():
            subset = train_df[train_df["category"] == category]
            # дедуп тот же, что у эмбеддингового классификатора: точные копии
            # с одинаковой меткой рвутся сплитом и завышают OOF
            subset = ProductQualityPredictor._dedup(subset).reset_index(drop=True)

            texts = build_model_texts(subset).values
            y = np.asarray(subset["label"].values, dtype=np.int64)
            n_pos = int(y.sum())
            if n_pos == 0 or n_pos == len(y):
                _log(f"{category}: один класс ({n_pos}/{len(y)}), категория пропущена")
                continue

            splits = max(2, min(n_splits, n_pos))
            if fam_by_id is not None:
                groups = np.asarray([fam_by_id.get(i, -1) for i in subset["id"]])
                cv = StratifiedGroupKFold(splits, shuffle=True, random_state=random_state)
                folds = list(cv.split(texts, y, groups))
            else:
                cv = StratifiedKFold(splits, shuffle=True, random_state=random_state)
                folds = list(cv.split(texts, y))

            # OOF для каждого векторизатора, порог подбирается на их среднем -
            # ровно та вероятность, что будет на инференсе
            oof = {kind: np.zeros(len(y), dtype=np.float64) for kind in _VECTORIZERS}
            for tr, va in folds:
                for kind in _VECTORIZERS:
                    vec = self._make_vectorizer(kind)
                    x_tr = _build_features(vec, texts[tr], subset.iloc[tr], fit=True)
                    x_va = _build_features(vec, texts[va], subset.iloc[va], fit=False)
                    clf = LogisticRegression(C=_C, max_iter=2000, class_weight="balanced")
                    clf.fit(x_tr, y[tr])
                    oof[kind][va] = clf.predict_proba(x_va)[:, 1]
            oof_mean = np.mean([oof[k] for k in _VECTORIZERS], axis=0)
            oof_f1, threshold = exact_best_threshold(oof_mean, y)
            if oof_f1 <= 0.0:
                raise RuntimeError(f"{category}: OOF F1 = 0, порог подбирать не на чем")

            heads = {}
            for kind in _VECTORIZERS:
                vec = self._make_vectorizer(kind)
                matrix = _build_features(vec, texts, subset, fit=True)
                clf = LogisticRegression(C=_C, max_iter=2000, class_weight="balanced")
                clf.fit(matrix, y)
                heads[kind] = {"vectorizer": vec, "classifier": clf}

            # retrieval-память: групповой OOF её вклад не видит по построению
            # (дубли всегда в одном фолде), включается как страховка - перенос
            # метки только при единогласии/чистоте, на train точность 0.9992
            memory = RetrievalMemory().fit(subset)

            self.category_models[category] = {
                "heads": heads,
                "memory": memory,
                "threshold": threshold,
                "C": _C,
                "oof_f1": oof_f1,
                "n_train": int(len(y)),
                "n_pos": n_pos,
            }
            report[category] = {
                k: v for k, v in self.category_models[category].items()
                if k not in ("heads", "memory")
            }
            _log(
                f"{category}: OOF F1={oof_f1:.4f} (ансамбль {'+'.join(_VECTORIZERS)}), "
                f"порог={threshold:.4f}, n={len(y)} (позитивов {n_pos}), {memory.summary()}"
            )

        if report:
            expected = train_df["category"].dropna().nunique()
            mean_f1 = sum(r["oof_f1"] for r in report.values()) / expected
            _log(f"ИТОГО средний OOF F1: {mean_f1:.4f}")
        return report

    @staticmethod
    def _make_vectorizer(kind: str = "word"):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if kind == "word":
            return TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=3,
                sublinear_tf=True,
                dtype=np.float32,
            )
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=300_000,
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
        memory_hits = 0
        for key, positions in groups.items():
            info = heads.get(key)
            if info is None:
                unknown += len(positions)
                continue
            index = np.asarray(positions, dtype=np.int64)
            sub = df.iloc[index]

            # 1) ансамбль: среднее вероятностей word- и char-голов
            block = np.zeros(len(index), dtype=np.float64)
            for head in info["heads"].values():
                matrix = _build_features(head["vectorizer"], texts[index], sub, fit=False)
                block += head["classifier"].predict_proba(matrix)[:, 1]
            block /= len(info["heads"])

            # 2) retrieval-память: точное совпадение с train перекрывает модель
            memory = info.get("memory")
            if memory is not None:
                try:
                    got, _src = memory.lookup(sub)
                    for j, g in enumerate(got):
                        if g is not None:
                            block[j] = 1.0 - 1e-5 if int(g) == 1 else 1e-5
                            memory_hits += 1
                except Exception as exc:
                    _log(f"память недоступна ({type(exc).__name__}: {exc}), продолжаю без неё")

            probs[index] = block
            preds[index] = (block >= float(info["threshold"])).astype(np.int64)

        # 3) absence veto: «топливо не входит в комплект» у легковоспламеняющихся ->
        # принудительный pred=0. На train: 1208 срабатываний, 0 убитых позитивов.
        try:
            veto = absence_veto_mask(df)
            if veto.any():
                probs[veto] = 1e-5
                preds[veto] = 0
                _log(f"veto применён к {int(veto.sum())} товарам")
        except Exception as exc:
            _log(f"veto недоступен ({type(exc).__name__}: {exc}), продолжаю без него")

        if memory_hits:
            _log(f"память перенесла вердикт для {memory_hits} товаров")
        if unknown:
            _log(f"{unknown} товаров с неизвестной категорией -> pred=0")
        return probs.tolist(), preds.tolist()

    def summary(self) -> str:
        if not self.category_models:
            return "текстовая модель пуста"
        parts = [
            f"{name}: порог {info['threshold']:.3f}, "
            f"голов {len(info['heads'])}, OOF {info.get('oof_f1', 0):.3f}"
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
            for key in ("heads", "threshold"):
                if key not in info:
                    raise ValueError(f"{name!r}: в артефакте нет поля {key!r}")
            for kind, head in info["heads"].items():
                for part in ("vectorizer", "classifier"):
                    if part not in head:
                        raise ValueError(f"{name!r}/{kind}: нет поля {part!r}")
            threshold = float(info["threshold"])
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name!r}: некорректный порог {threshold}")


# Обучение из командной строки:
#   venv312/bin/python -m src.text_model data.csv text_model.joblib
def _main(argv: Sequence[str]) -> None:
    if len(argv) not in (2, 3):
        raise SystemExit(
            "использование: python -m src.text_model <data.csv> <выходной.joblib> [families.csv]"
        )
    # при запуске через -m этот модуль называется __main__, и класс запиклился
    # бы как __main__.TextQualityModel - такой артефакт не загрузится из run.py.
    # Импортируем класс под каноническим именем модуля.
    from src.text_model import TextQualityModel as CanonicalTextQualityModel

    data_path, out_path = argv[0], argv[1]
    families = pd.read_csv(argv[2]) if len(argv) == 3 else None
    # prepare_dataframe, а не голый read_csv: без image_paths память
    # не построит SHA/dHash-ключи и будет работать только по тексту
    from pathlib import Path

    from src.utils_data_prep import prepare_dataframe

    data_path = Path(data_path)
    df = prepare_dataframe(data_path, data_path.parent / "images")
    model = CanonicalTextQualityModel()
    model.train(df, families=families)
    model.save(out_path)
    _log(f"артефакт сохранён: {out_path}")


if __name__ == "__main__":
    _main(sys.argv[1:])
