"""Точка входа submit-решения E-CUP «Контроль качества».

Главный инвариант: процесс обязан записать валидный CSV со строкой на каждый
входной id. Падение на любом наборе означает ноль за весь прогон и остановку
проверки на остальных наборах, поэтому тяжёлые стадии обёрнуты в try/except
и деградируют, а не роняют решение.
"""

import argparse
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
IMAGE_RETRIEVAL_PATH = ROOT / "image_retrieval.npz"

_SHARED_MODELS_DIR = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
MODEL_EMBED_PATH = _SHARED_MODELS_DIR / "Qwen/Qwen3-VL-Embedding-2B"
MODEL_LLM_PATH = _SHARED_MODELS_DIR / "Qwen/Qwen3.5-4B"


def _log(message: str) -> None:
    print(f"[run] {message}", file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product quality predictor submit pipeline")
    # описание организаторов и baseline расходятся в написании, принимаем оба
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

    # Шаги 2-3: классификация. Основной путь - текстовая модель (OOF 0.871,
    # секунды на CPU). Эмбеддинговый путь (0.505 на лидерборде) остаётся
    # запасным: он включается, только если текстовый артефакт не отработал.
    classified = False
    try:
        from src.text_model import TextQualityModel

        text_model = TextQualityModel.load(str(TEXT_MODEL_PATH))
        _log(f"текстовая модель: {text_model.summary()}")
        current_df["logreg_prob"], current_df["pred"] = text_model.predict(current_df)
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
            current_df["embedding"] = [[0.0] * 2048] * n
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
    try:
        comments = _generate(current_df)
    except Exception as exc:
        _log(f"комментарии не сгенерированы ({type(exc).__name__}: {exc}), берём заглушки")
        traceback.print_exc(file=sys.stderr)
        comments = []

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
