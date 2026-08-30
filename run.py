"""Точка входа submit-решения E-CUP «Контроль качества».

Главный инвариант: процесс обязан записать валидный CSV со строкой на каждый
входной id. Падение на любом наборе означает ноль за весь прогон и остановку
проверки на остальных наборах, поэтому тяжёлые стадии обёрнуты в try/except
и деградируют, а не роняют решение.
"""

import argparse
import gc
import math
import os
import sys
import traceback
from pathlib import Path

# проверка идёт офлайн; ставим ДО импорта transformers, иначе он успеет
# сходить в сеть за конфигом и повиснуть на таймауте
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from src.constants import PIXEL_PRESETS, DEFAULT_EMBED_BATCH_SIZE, DEFAULT_LLM_BATCH_SIZE
from src.output_validation import validate_csv_file
from src.utils_data_prep import prepare_dataframe
from src.utils_logreg import ProductQualityPredictor
from src.utils_postprocess import format_results

# пути только от файла: рабочий каталог в контейнере не гарантирован
ROOT = Path(__file__).resolve().parent
CLASSIFIER_PATH = ROOT / "baseline_qwen3vl_bf16.joblib"
TEXT_MODEL_PATH = ROOT / "text_model.joblib"
E5_MODEL_DIR = ROOT / "models" / "multilingual-e5-base"
IMAGE_RETRIEVAL_PATH = ROOT / "image_retrieval.npz"
TEMPLATE_TRANSFER_PATH = ROOT / "template_transfer.json"
LORA_ADAPTER_DIR = ROOT / "qc_lora_adapter"
LORA_BLEND_PATH = ROOT / "lora_blend.json"

_SHARED_MODELS_DIR = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
MODEL_EMBED_PATH = _SHARED_MODELS_DIR / "Qwen/Qwen3-VL-Embedding-2B"
MODEL_LLM_PATH = _SHARED_MODELS_DIR / "Qwen/Qwen3.5-4B"


def _log(message: str) -> None:
    print(f"[run] {message}", file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product quality predictor submit pipeline")
    parser.add_argument("--test_data_path", "--test-data-path", "-i",
                        dest="test_data_path", required=True, type=Path)
    parser.add_argument("--output_path", "--output-path", "-o",
                        dest="output_path", required=True, type=Path)
    return parser.parse_args()


def _write_csv(output_path: Path, ids, results) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "result": results}).to_csv(output_path, index=False)


# Аварийный выход: валидный CSV с заглушками на все id. Плохая метрика
# несравнимо лучше нуля за прогон и остановки проверки.
def _write_fallback(output_path: Path, data_path: Path) -> None:
    try:
        ids = pd.read_csv(data_path, usecols=["id"])["id"].tolist()
    except Exception:
        _log("не удалось прочитать даже id из входного файла")
        return
    results = format_results([], [], n=len(ids))
    _write_csv(output_path, ids, results)
    validate_csv_file(output_path, ids)
    _log(f"записан аварийный CSV на {len(ids)} строк")


def _embed(current_df):
    from src.utils_embed_cuda import embed_data_cuda

    return embed_data_cuda(
        str(MODEL_EMBED_PATH), current_df,
        max_pixels=PIXEL_PRESETS["M"],
        batch_size=DEFAULT_EMBED_BATCH_SIZE,
    )


def _generate(current_df):
    from src.utils_generate_cuda import generate_comments_cuda

    return generate_comments_cuda(
        str(MODEL_LLM_PATH), current_df,
        batch_size=DEFAULT_LLM_BATCH_SIZE,
    )


def _apply_retrieval(current_df):
    from src.image_retrieval import (
        ImageRetrievalIndex,
        apply_image_retrieval,
        load_retrieval_index,
    )

    retrieval = ImageRetrievalIndex(load_retrieval_index(IMAGE_RETRIEVAL_PATH))
    result = apply_image_retrieval(
        current_df,
        current_df["logreg_prob"].tolist(),
        current_df["pred"].tolist(),
        retrieval,
    )
    current_df["logreg_prob"] = result["probabilities"]
    current_df["pred"] = result["predictions"]
    current_df["retrieval_reason"] = result["reasons"]
    return result


def main() -> None:
    args = _parse_args()
    data_path = Path(args.test_data_path)
    output_path = Path(args.output_path)

    # Шаг 1: текст и пути к картинкам
    images_path = data_path.parent / "images"
    current_df = prepare_dataframe(data_path, images_path)
    n = len(current_df)
    _log(f"товаров {n}, картинок {int(current_df['n_images'].sum())}, "
         f"без картинок {int((current_df['n_images'] == 0).sum())}")

    # Шаги 2-3: классификация. Основной путь - текстовая модель (family-OOF
    # около 0.86, секунды на CPU). Эмбеддинговый путь (0.505 на лидерборде) остаётся
    # запасным: он включается, только если текстовый артефакт не отработал.
    classified = False
    try:
        from src.text_model import TextQualityModel

        text_model = TextQualityModel.load(str(TEXT_MODEL_PATH))
        _log(f"текстовая модель: {text_model.summary()}")
        e5_path = str(E5_MODEL_DIR) if E5_MODEL_DIR.exists() else None
        current_df["logreg_prob"], current_df["pred"] = text_model.predict(
            current_df, e5_model_path=e5_path
        )
        classified = True
    except Exception as exc:
        _log(f"текстовая модель недоступна ({type(exc).__name__}: {exc}), "
             "переключаюсь на эмбеддинги")
        traceback.print_exc(file=sys.stderr)

    if not classified:
        try:
            embeddings = _embed(current_df)
            current_df["embedding"] = list(embeddings)
        except Exception as exc:
            _log(f"эмбеддинги не посчитаны ({type(exc).__name__}: {exc}), идём с нулями")
            traceback.print_exc(file=sys.stderr)
            from src.constants import EMBEDDING_DIM

            current_df["embedding"] = [[0.0] * EMBEDDING_DIM] * n
        try:
            predictor = ProductQualityPredictor.load(str(CLASSIFIER_PATH))
            _log(f"классификатор: {predictor.summary()}")
            current_df["logreg_prob"], current_df["pred"] = predictor.predict(
                current_df["embedding"], current_df["category"]
            )
        except Exception as exc:
            _log(f"классификатор недоступен ({type(exc).__name__}: {exc}), все pred=0")
            traceback.print_exc(file=sys.stderr)
            current_df["logreg_prob"] = [0.0] * n
            current_df["pred"] = [0] * n

    # Шаг 3.7: LoRA-VLM ансамбль - строго ДО retrieval, чтобы
    # высокоточные image-overrides оставались финальным словом.
    if classified and LORA_ADAPTER_DIR.is_dir():
        try:
            from src.lora_classifier import LoraClassifier, load_blend_config

            blend = load_blend_config(LORA_BLEND_PATH)
            weights = blend.get("weights", {})
            thresholds = blend.get("thresholds", {})
            if weights:
                _log("загружаю LoRA-VLM для ансамбля")
                classifier = LoraClassifier(
                    str(MODEL_LLM_PATH), str(LORA_ADAPTER_DIR)
                )
                vlm_probs = classifier.predict_proba(current_df)
                text_probs = current_df["logreg_prob"].tolist()
                categories = current_df["category"].astype(str).tolist()
                final_probs, final_preds = [], []
                # Смешиваем в пространстве ЛОГИТОВ, а не вероятностей: все наши
                # замеры сделаны так, и линейное усреднение даёт другой ответ -
                # оно тянет результат к середине и гасит уверенность канала,
                # который прав. Вес - доля LoRA (0 = только текст, 1 = только VLM),
                # подобран на четырёх фолдах с проверкой на пятом.
                def _logit(value: float) -> float:
                    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
                    return math.log(value / (1.0 - value))

                def _norm_key(value: object) -> str:
                    # та же нормализация, что в _norm_category: ё и лишние
                    # пробелы не должны разваливать сопоставление с конфигом
                    return " ".join(str(value).lower().replace("ё", "е").split())

                weights = {_norm_key(k): v for k, v in weights.items()}
                thresholds = {_norm_key(k): v for k, v in thresholds.items()}
                brand_weights = {_norm_key(k): v
                                 for k, v in blend.get("brand_weights", {}).items()}
                # бренд-канал: статистика первого токена названия (см.
                # src/brand_channel.py); без артефакта канал молча выключен
                brand_stats = None
                if brand_weights and any(float(v) > 0 for v in brand_weights.values()):
                    try:
                        from src.brand_channel import brand_prob, load_stats

                        brand_stats = load_stats(ROOT / "brand_stats.json")
                        _log("бренд-канал подключён")
                    except Exception as exc:
                        _log(f"бренд-канал недоступен ({type(exc).__name__}: {exc})")
                        brand_stats = None
                names = current_df["name"].tolist()
                text_preds = current_df["pred"].tolist()
                skipped = 0
                for text_prob, text_pred, vlm_prob, category, name in zip(
                        text_probs, text_preds, vlm_probs, categories, names):
                    key = _norm_key(category)
                    # нет веса/порога для категории или VLM-батч упал (None):
                    # оставляем решение текстовой модели с ЕЁ калиброванным
                    # порогом, а не перепороживаем на 0.5
                    if key not in weights or key not in thresholds or vlm_prob is None:
                        final_probs.append(float(text_prob))
                        final_preds.append(int(text_pred))
                        skipped += 1
                        continue
                    w = float(weights[key])
                    wb = float(brand_weights.get(key, 0.0)) if brand_stats else 0.0
                    mixed_logit = ((1.0 - w - wb) * _logit(text_prob)
                                   + w * _logit(vlm_prob))
                    if wb > 0:
                        mixed_logit += wb * _logit(brand_prob(brand_stats, category, name))
                    blended = 1.0 / (1.0 + math.exp(-mixed_logit))
                    final_probs.append(blended)
                    final_preds.append(int(blended >= float(thresholds[key])))
                current_df["logreg_prob"] = final_probs
                current_df["pred"] = final_preds
                positives = int(sum(final_preds))
                _log(f"ансамбль применён: pred=1 у {positives} из {n}"
                     + (f", без смеси {skipped}" if skipped else ""))
                del classifier
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                _log("lora_blend.json пуст или не читается - ансамбль выключен")
        except Exception as exc:
            _log(f"LoRA-ансамбль недоступен ({type(exc).__name__}: {exc}), остаётся текстовая модель")
            traceback.print_exc(file=sys.stderr)

    # Шаг 3.8: перенос меток по шаблонам продавца (LOO-точность 99%).
    # ДО retrieval: точные картиночные совпадения сохраняют последнее слово.
    # Если тест не делит шаблоны с train - слой молчит и ничего не меняет.
    try:
        from src.template_transfer import apply_transfer, load_index

        transfer_index = load_index(TEMPLATE_TRANSFER_PATH)
        probs_t, preds_t, reasons_t = apply_transfer(
            current_df,
            current_df["logreg_prob"].tolist(),
            current_df["pred"].tolist(),
            transfer_index,
        )
        current_df["logreg_prob"] = probs_t
        current_df["pred"] = preds_t
    except Exception as exc:
        _log(f"шаблонный перенос недоступен ({type(exc).__name__}: {exc})")

    # Шаг 3.85: kNN-перенос по эмбеддингам (почти-дубли train-карточек,
    # которые точные ключи не видят). Без артефактов слой молча выключен.
    try:
        knn_npz = ROOT / "knn_transfer.npz"
        knn_cfg = ROOT / "knn_transfer.json"
        if knn_npz.is_file() and knn_cfg.is_file():
            from src.knn_transfer import apply_knn_transfer, load_index as load_knn

            knn_index = load_knn(knn_npz, knn_cfg)
            probs_k, preds_k, _ = apply_knn_transfer(
                current_df,
                current_df["logreg_prob"].tolist(),
                current_df["pred"].tolist(),
                knn_index,
                str(MODEL_EMBED_PATH),
            )
            current_df["logreg_prob"] = probs_k
            current_df["pred"] = preds_k
    except Exception as exc:
        _log(f"kNN-перенос недоступен ({type(exc).__name__}: {exc})")

    try:
        retrieval_result = _apply_retrieval(current_df)
        changed = int((retrieval_result["reasons"] != "text_model").sum())
        _log(
            f"image retrieval: backend={retrieval_result['backend']}, "
            f"решений={changed}"
        )
    except Exception as exc:
        _log(f"image retrieval недоступен ({type(exc).__name__}: {exc}), оставляем text pred")
        traceback.print_exc(file=sys.stderr)

    positives = int(sum(current_df["pred"]))
    _log(f"pred=1 у {positives} из {n}")

    # Шаг 4: комментарии. Без LLM останутся заглушки нужной длины.
    # stub_comments.flag - диагностический сабмит: те же вердикты, заглушки
    # вместо LLM (замер, участвуют ли комментарии в автометрике LB).
    if (ROOT / "stub_comments.flag").exists():
        _log("stub_comments.flag: генерация выключена, идут заглушки")
        comments = []
    else:
        try:
            comments = _generate(current_df)
        except Exception as exc:
            _log(f"комментарии не сгенерированы ({type(exc).__name__}: {exc}), берём заглушки")
            traceback.print_exc(file=sys.stderr)
            comments = []
        # брак LLM (пустота, слова «бан», логическая инверсия) заменяется
        # детерминированной формулировкой из флагов правил
        try:
            from src.comment_fallback import repair_comments

            comments = repair_comments(comments, current_df)
        except Exception as exc:
            _log(f"фолбэк комментариев недоступен ({type(exc).__name__}: {exc})")

    # Шаг 5: сборка строк ровно по числу товаров
    results = format_results(comments, current_df["pred"].tolist(), n=n)

    # Шаг 6: запись и строгая проверка контракта
    ids = current_df["id"].tolist()
    _write_csv(output_path, ids, results)
    validate_csv_file(output_path, ids)
    _log(f"готово: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        # последняя попытка отдать валидный файл
        try:
            args = _parse_args()
            _write_fallback(Path(args.output_path), Path(args.test_data_path))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
