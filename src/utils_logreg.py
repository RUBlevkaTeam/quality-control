"""Классификатор качества товара: своя линейная голова на каждую категорию.

Инференс никогда не распаковывает pickle - joblib нужен только в save()/load().
Логистическую регрессию predict() считает через СВЕЖИЙ sklearn.LogisticRegression
с вручную подставленными coef_/intercept_/classes_: это ровно тот же результат,
что и predict_proba() исходной модели (проверено побитово), но без unpickle -
код всегда выполняется в той версии sklearn, что реально стоит в контейнере,
а не в той, что была при обучении. Смена sklearn 1.6->1.7 убрала атрибут
multi_class, которого при unpickle старого объекта не будет - этот риск снят.
Если sklearn в контейнере всё же недоступен, predict() откатывается на
хендролленную numpy-сигмоиду (тот же результат, проверено).

PCA - исключение: sklearn.PCA.transform() безусловно читает приватный
атрибут explained_variance_, даже когда whiten=False - это деталь реализации,
не часть документированного контракта fitted-атрибутов, и в другой версии
sklearn список нужных атрибутов может незаметно измениться. Поэтому PCA-проекция
всегда считается вручную: (X - mean_) @ components_.T - это математическое
определение проекции, оно не может измениться в новой версии библиотеки.

Класс намеренно сохраняет имя и путь модуля: артефакт baseline_qwen3vl_bf16.joblib
запиклен как src.utils_logreg.ProductQualityPredictor и без этого не распакуется.
"""
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# регулярки
_WS_RE = re.compile(r"\s+")
_YO_MAP = str.maketrans({"ё": "е", "Ё": "Е"})

# массив из 99 порогов, по которым перебирается F1 при подборе точки классификатора
_THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

_NPZ_FORMAT_VERSION = 2


# приводит имя категории к каноничному виду, используя регулярки выше
def _norm_category(value: Any) -> str:
    if value is None or value != value:  # None или NaN
        return ""
    text = str(value).translate(_YO_MAP).strip().casefold() # ё -> е, обрезаем пробелы и приводим к нижнему регистру
    return _WS_RE.sub(" ", text)


# Строит НЕОБУЧЕННЫЙ LogisticRegression() и вручную подставляет ему
# готовые веса - вместо unpickle исходной модели. decision_function()
# у sklearn читает только coef_/intercept_/classes_ (проверено по исходникам
# LinearClassifierMixin), это стабильный публичный контракт, а не приватная
# деталь. Возвращает None, если sklearn недоступен - тогда predict() посчитает
# ту же сигмоиду руками.
def _build_fresh_logreg(w: np.ndarray, b: float):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    clf = LogisticRegression()
    clf.coef_ = w.reshape(1, -1)
    clf.intercept_ = np.array([b], dtype=np.float64)
    clf.classes_ = np.array([0, 1])
    clf.n_features_in_ = w.shape[0]
    return clf


class ProductQualityPredictor:
    def __init__(self) -> None:
        # {оригинальное имя категории: {'model', 'threshold', ...}}
        self.category_models: Dict[str, dict] = {}

    # кэш линейного представления не должен попадать в pickle
    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "_linear"}

    # ------------------------------------------------------------------
    # линейное представление
    # ------------------------------------------------------------------

    # {нормализованная категория: {name, w, b, threshold, normalize, pca_mean, pca_components}}
    # строится лениво из category_models либо приходит готовым из from_npz
    def _heads(self) -> Dict[str, dict]:
        cached = getattr(self, "_linear", None)
        if cached is not None:
            return cached

        heads: Dict[str, dict] = {}
        for name, info in getattr(self, "category_models", {}).items():
            key = _norm_category(name)
            heads[key] = self._head_from_estimator(
                name=name,
                estimator=info["model"],
                threshold=float(info["threshold"]),
                normalize=bool(info.get("normalize", False)),
            )
        self._linear = heads
        return heads

    # Разбирает обученную модель на голые числа. Принимает и одиночный
    # LogisticRegression, и Pipeline со ступенью PCA - во втором случае
    # матрица проекции сохраняется отдельно, потому что L2-нормировка
    # между шагами нелинейна и свернуть всё в одни веса нельзя.
    @staticmethod
    def _head_from_estimator(name, estimator, threshold: float, normalize: bool) -> dict:
        pca_mean = None
        pca_components = None
        clf = estimator

        steps = getattr(estimator, "named_steps", None)
        if steps is not None:
            clf = steps["clf"]
            pca = steps.get("pca")
            if pca is not None:
                pca_mean = np.asarray(pca.mean_, dtype=np.float64)
                pca_components = np.asarray(pca.components_, dtype=np.float64)

        w = np.asarray(clf.coef_[0], dtype=np.float64)
        b = float(clf.intercept_[0])
        return {
            "name": name,
            "w": w,
            "b": b,
            "threshold": threshold,
            "normalize": normalize,
            "pca_mean": pca_mean,
            "pca_components": pca_components,
            "sk_clf": _build_fresh_logreg(w, b),
        }

    # Повторяет ступени обучающего Pipeline в том же порядке:
    # L2-нормировка -> центрирование и проекция PCA.
    @staticmethod
    def _apply_transforms(block: np.ndarray, head: dict) -> np.ndarray:
        if head["normalize"]:
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            block = block / np.maximum(norms, 1e-12)
        components = head.get("pca_components")
        if components is not None:
            block = (block - head["pca_mean"]) @ components.T
        return block

    # сколько признаков голова ждёт на ВХОДЕ (до сжатия)
    @staticmethod
    def _input_dim(head: dict) -> int:
        components = head.get("pca_components")
        if components is not None:
            return int(components.shape[1])
        return int(head["w"].shape[0])

    # эмбеддинги -> матрица (n, d) или None, если вход непригоден
    @staticmethod
    def _as_matrix(embeddings: Iterable, n: int) -> np.ndarray | None:
        source = embeddings.to_numpy() if hasattr(embeddings, "to_numpy") else embeddings

        # быстрый путь: одна конвертация на C вместо цикла по 3800 спискам
        matrix = None
        if isinstance(source, np.ndarray) and source.ndim == 2 and source.dtype != object:
            matrix = source.astype(np.float64, copy=False)
        else:
            if not isinstance(source, (list, tuple, np.ndarray)):
                source = list(source)
            try:
                candidate = np.asarray(source if len(source) else [], dtype=np.float64)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and candidate.ndim == 2:
                matrix = candidate

        # медленный путь
        if matrix is None:
            try:
                rows = [np.asarray(item, dtype=np.float64).ravel() for item in source]
            except (TypeError, ValueError):
                return None
            if not rows:
                return None
            widths = {row.shape[0] for row in rows}
            if len(widths) != 1:
                return None
            matrix = np.stack(rows)

        if matrix.shape[0] != n:
            return None
        return matrix

    # ------------------------------------------------------------------
    # инференс
    # ------------------------------------------------------------------

    # Возвращает (вероятности, решения) - по одному значению на товар.
    # Никогда не бросает исключение: любая проблема со входом деградирует
    # до prob=0.0 / pred=0, потому что упавший прогон стоит дороже,
    # чем неверный вердикт по части товаров.
    def predict(self, embeddings: Iterable, categories: Iterable) -> Tuple[List[float], List[int]]:
        raw_cats = categories.tolist() if hasattr(categories, "tolist") else list(categories)
        n = len(raw_cats)
        probs = np.zeros(n, dtype=np.float64)
        preds = np.zeros(n, dtype=np.int64)
        if n == 0:
            return [], []

        heads = self._heads()
        if not heads:
            return probs.tolist(), preds.tolist()

        matrix = self._as_matrix(embeddings, n)
        if matrix is None:
            return probs.tolist(), preds.tolist()

        finite = np.isfinite(matrix).all(axis=1)

        groups: Dict[str, List[int]] = {}
        for i, raw in enumerate(raw_cats):
            groups.setdefault(_norm_category(raw), []).append(i)

        for key, positions in groups.items():
            index = np.asarray(positions, dtype=np.int64)
            index = index[finite[index]]
            if index.size == 0:
                continue
            head = heads.get(key)
            if head is None:
                continue
            expected = self._input_dim(head)
            if matrix.shape[1] != expected:
                continue

            block = self._apply_transforms(matrix[index], head)
            sk_clf = head.get("sk_clf")
            if sk_clf is not None:
                # тот же predict_proba, что и у исходной модели - просто
                # без прохода через unpickle
                block_probs = sk_clf.predict_proba(block)[:, 1]
            else:
                # sklearn недоступен: та же арифметика руками, устойчивая
                # к переполнению на больших по модулю логитах
                logits = block @ head["w"] + head["b"]
                block_probs = np.where(
                    logits >= 0,
                    1.0 / (1.0 + np.exp(-np.abs(logits))),
                    np.exp(-np.abs(logits)) / (1.0 + np.exp(-np.abs(logits))),
                )
            probs[index] = block_probs
            preds[index] = (block_probs >= head["threshold"]).astype(np.int64)

        return probs.tolist(), preds.tolist()

    # компактная сводка для логов прогона
    def summary(self) -> str:
        heads = self._heads()
        if not heads:
            return "предиктор пуст"
        parts = []
        for h in heads.values():
            piece = f"{h['name']}: порог {h['threshold']:.2f}, вход {self._input_dim(h)}"
            if h.get("pca_components") is not None:
                piece += f" -> PCA {h['w'].shape[0]}"
            if h["normalize"]:
                piece += ", L2-норм."
            parts.append(piece)
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # обучение
    # ------------------------------------------------------------------

    # Порог подбирается на out-of-fold вероятностях той же конфигурации,
    # которая уходит в финальную модель. У baseline порог снимался с модели,
    # обученной на 80%, а в артефакт клалась модель, обученная на 100%.
    def train(
        self,
        train_df,
        *,
        c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        normalize_grid: Sequence[bool] = (False, True),
        pca_grid: Sequence[Any] = (None,),
        n_splits: int = 5,
        dedup: bool = True,
        random_state: int = 42,
        n_trials: int = 0,
    ) -> Dict[str, dict]:
        self.category_models = {}
        self.__dict__.pop("_linear", None)
        report: Dict[str, dict] = {}

        for category in train_df["category"].dropna().unique():
            subset = train_df[train_df["category"] == category]
            if dedup:
                subset = self._dedup(subset)

            X = np.vstack([np.asarray(e, dtype=np.float64).ravel() for e in subset["embedding"]])
            y = np.asarray(subset["label"].values, dtype=np.int64)
            n_pos = int(y.sum())
            if n_pos == 0 or n_pos == len(y):
                continue

            # с 198 позитивами на 5 фолдов в каждом остаётся ~40 штук;
            # меньше 2 позитивов на фолд StratifiedKFold не разложит
            splits = max(2, min(n_splits, n_pos))

            # ГИПЕРПАРАМЕТРЫ
            evaluate = self._make_objective(X, y, splits, random_state)
            if n_trials:
                best = self._search_optuna(
                    evaluate, X, y, splits, c_grid, normalize_grid,
                    pca_grid, n_trials, random_state,
                )
            else:
                best = self._search_grid(evaluate, c_grid, normalize_grid, pca_grid)

            if best is None or best["oof_f1"] <= 0.0:
                # у baseline в этом случае молча оставался порог 0.5
                raise RuntimeError(
                    f"{category}: ни одна конфигурация не дала F1 > 0 - "
                    "эмбеддинги или метки не несут сигнала, порог подбирать не на чем"
                )

            final = self._build_estimator(best["C"], best["normalize"], best["pca"], random_state)
            final.fit(X, y)

            self.category_models[category] = {
                "model": final,
                "threshold": best["threshold"],
                "normalize": best["normalize"],
                "C": best["C"],
                "pca": best["pca"],
                "oof_f1": best["oof_f1"],
                "n_train": int(len(y)),
                "n_pos": n_pos,
                "dim": int(X.shape[1]),
            }
            report[category] = {k: v for k, v in self.category_models[category].items() if k != "model"}

        return report

    # Ступени идут ровно в том же порядке, что и _apply_transforms в инференсе.
    # PCA стоит ВНУТРИ пайплайна: обученный заранее на всей выборке, он подсмотрел бы
    # валидационные фолды и завысил OOF-оценку.
    @staticmethod
    def _build_estimator(C: float, normalize: bool, n_components, random_state: int):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer

        steps = []
        if normalize:
            steps.append(("norm", Normalizer()))
        if n_components:
            steps.append(("pca", PCA(n_components=int(n_components), random_state=random_state)))
        steps.append(("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced")))
        return Pipeline(steps)

    # (C, normalize, pca) -> (OOF F1, порог). Одна точка пространства поиска.
    def _make_objective(self, X, y, splits: int, random_state: int):
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        cache: Dict[tuple, tuple] = {}

        def evaluate(C, normalize, pca) -> Tuple[float, float]:
            key = (round(float(C), 10), bool(normalize), pca)
            if key not in cache:
                cv = StratifiedKFold(splits, shuffle=True, random_state=random_state)
                oof = cross_val_predict(
                    self._build_estimator(C, normalize, pca, random_state),
                    X, y, cv=cv, method="predict_proba",
                )[:, 1]
                cache[key] = self._best_threshold(oof, y, f1_score)
            return cache[key]

        return evaluate

    # PCA не выдаст больше компонент, чем строк в обучающей части фолда
    @staticmethod
    def _max_components(X: np.ndarray, splits: int) -> int:
        rows_in_fold = int(len(X) * (splits - 1) / splits)
        return max(2, min(X.shape[1], rows_in_fold - 1))

    @staticmethod
    def _search_grid(evaluate, c_grid, normalize_grid, pca_grid):
        best = None
        for normalize in normalize_grid:
            for pca in pca_grid:
                for C in c_grid:
                    f1, threshold = evaluate(C, normalize, pca)
                    if best is None or f1 > best["oof_f1"]:
                        best = {"oof_f1": f1, "threshold": threshold,
                                "C": float(C), "normalize": bool(normalize), "pca": pca}
        return best

    # optuna опциональна: без неё откатываемся на полный перебор, чтобы
    # отсутствие dev-зависимости не ломало обучение
    def _search_optuna(self, evaluate, X, y, splits, c_grid, normalize_grid,
                       pca_grid, n_trials, random_state):
        try:
            import optuna
        except ImportError:
            return self._search_grid(evaluate, c_grid, normalize_grid, pca_grid)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        c_low, c_high = float(min(c_grid)), float(max(c_grid))
        cap = self._max_components(X, splits)
        # ниже 8 компонент сжимать нечего, а cap<16 означает крошечную выборку
        allow_pca = cap >= 16

        def objective(trial):
            C = trial.suggest_float("C", c_low, c_high, log=True)
            normalize = trial.suggest_categorical("normalize", [False, True])
            pca = None
            if allow_pca and trial.suggest_categorical("use_pca", [False, True]):
                pca = trial.suggest_int("n_components", 8, cap, log=True)
            f1, threshold = evaluate(C, normalize, pca)
            trial.set_user_attr("threshold", threshold)
            trial.set_user_attr("pca", pca)
            return f1

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        # стартуем с текущей ручной сетки, чтобы поиск гарантированно не был хуже неё
        for normalize in normalize_grid:
            for C in c_grid:
                study.enqueue_trial(
                    {"C": float(C), "normalize": bool(normalize), "use_pca": False},
                    skip_if_exists=True,
                )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        trial = study.best_trial
        return {
            "oof_f1": float(trial.value),
            "threshold": float(trial.user_attrs["threshold"]),
            "C": float(trial.params["C"]),
            "normalize": bool(trial.params["normalize"]),
            "pca": trial.user_attrs["pca"],
        }

    # Дубли по name+description рвутся сплитом между train и val и завышают оценку.
    # Схлопываем только полные копии С ОДИНАКОВОЙ меткой: если дедуплицировать по
    # одному тексту, группа с противоречивой разметкой потеряет целый класс.
    @staticmethod
    def _dedup(subset):
        columns = [c for c in ("name", "description") if c in subset.columns]
        if not columns:
            return subset

        key = None
        for column in columns:
            part = subset[column].fillna("").astype(str).str.lower().str.replace(
                r"[^0-9a-zа-яё]+", "", regex=True
            )
            key = part if key is None else key + "||" + part

        keep = ~key.to_frame("key").assign(label=subset["label"].values).duplicated()
        return subset[keep.values]

    # Возвращает (лучший F1, порог из СЕРЕДИНЫ плато). Baseline брал первый
    # максимум из-за строгого '>', то есть самый агрессивный порог из равных.
    @staticmethod
    def _best_threshold(probs: np.ndarray, y: np.ndarray, f1_score) -> Tuple[float, float]:
        scores = np.array([f1_score(y, (probs >= t).astype(int)) for t in _THRESHOLD_GRID])
        best = float(scores.max())
        if best <= 0.0:
            return 0.0, 0.5
        plateau = _THRESHOLD_GRID[scores >= best - 1e-12]
        return best, float(np.median(plateau))

    # ------------------------------------------------------------------
    # сериализация
    # ------------------------------------------------------------------

    # Формат без pickle: не зависит ни от версии sklearn, ни от того, как
    # называется этот модуль. Именно это стоит класть в submit-архив.
    def export_npz(self, filepath) -> None:
        heads = self._heads()
        if not heads:
            raise ValueError("нечего экспортировать: предиктор пуст")
        ordered = [heads[key] for key in sorted(heads)]

        # сверяем размер ВХОДА (эмбеддинга), а не весов: после сжатия PCA
        # веса у категорий короче входа и могут быть разной длины
        input_dims = {self._input_dim(h) for h in ordered}
        if len(input_dims) != 1:
            raise ValueError(f"категории ждут разное число признаков на входе: {sorted(input_dims)}")

        arrays: Dict[str, np.ndarray] = {
            "format_version": np.array(_NPZ_FORMAT_VERSION),
            "categories": np.array([h["name"] for h in ordered]),
            "intercept": np.array([h["b"] for h in ordered], dtype=np.float64), # смещение
            "thresholds": np.array([h["threshold"] for h in ordered], dtype=np.float64),
            "normalize": np.array([h["normalize"] for h in ordered], dtype=bool),
        }
        # веса - отдельным массивом на категорию: одной матрицей их уже не сложить
        for i, head in enumerate(ordered):
            arrays[f"coef_{i}"] = head["w"] # ВЕСА
            if head.get("pca_components") is not None:
                arrays[f"pca_components_{i}"] = head["pca_components"]
                arrays[f"pca_mean_{i}"] = head["pca_mean"]

        np.savez(filepath, **arrays)

    @classmethod
    def from_npz(cls, filepath) -> "ProductQualityPredictor":
        data = np.load(filepath, allow_pickle=False)
        version = int(data["format_version"])
        if version != _NPZ_FORMAT_VERSION:
            raise ValueError(f"неизвестная версия формата: {version}")

        predictor = cls()
        heads: Dict[str, dict] = {}
        for i, name in enumerate(data["categories"].tolist()):
            has_pca = f"pca_components_{i}" in data.files
            w = np.asarray(data[f"coef_{i}"], dtype=np.float64)
            b = float(data["intercept"][i])
            heads[_norm_category(name)] = {
                "name": name,
                "w": w,
                "b": b,
                "threshold": float(data["thresholds"][i]),
                "normalize": bool(data["normalize"][i]),
                "pca_components": np.asarray(data[f"pca_components_{i}"], dtype=np.float64) if has_pca else None,
                "pca_mean": np.asarray(data[f"pca_mean_{i}"], dtype=np.float64) if has_pca else None,
                "sk_clf": _build_fresh_logreg(w, b),
            }
        predictor._linear = heads
        return predictor

    def save(self, filepath) -> None:
        import joblib

        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        import joblib

        return joblib.load(filepath)
