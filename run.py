import argparse
import os
from pathlib import Path

from src.constants import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_LOCAL_EMBED_BATCH_SIZE,
    DEFAULT_LOCAL_LLM_BATCH_SIZE,
    DEFAULT_LLM_BATCH_SIZE,
    DEFAULT_PIXEL_PRESET,
    DEFAULT_SHARED_MODELS_PATH,
    EMBED_MODEL_NAME,
    LLM_MODEL_NAME,
    PIXEL_PRESETS,
)
from src.utils_data_prep import prepare_dataframe
from src.utils_logreg import ProductQualityPredictor
from src.utils_postprocess import format_results

PROJECT_DIR = Path(__file__).resolve().parent
CLASSIFIER_JOBLIB_PATH = PROJECT_DIR / "baseline_qwen3vl_bf16.joblib"
CLASSIFIER_NPZ_PATH = PROJECT_DIR / "baseline_qwen3vl_bf16.npz"


def resolve_models_dir(explicit_path: Path | None = None) -> Path:
    """Use an explicit path, the evaluator mount, or the local models directory."""
    if explicit_path is not None:
        return explicit_path
    shared_models_path = os.environ.get("SHARED_MODELS_PATH")
    if shared_models_path:
        return Path(shared_models_path)
    local_models_path = PROJECT_DIR / "models"
    if local_models_path.is_dir():
        return local_models_path
    return Path(DEFAULT_SHARED_MODELS_PATH)


def resolve_classifier_path() -> Path:
    """Prefer the portable linear NPZ, while keeping old joblib submissions valid."""
    if CLASSIFIER_NPZ_PATH.is_file():
        return CLASSIFIER_NPZ_PATH
    return CLASSIFIER_JOBLIB_PATH


def main() -> None:

    # передаем из терминала расположение тестовых данных и выхода как пути
    parser = argparse.ArgumentParser(description="Product quality predictor submit pipeline")
    parser.add_argument(
        "-i",
        "--test_data_path",
        "--test-data-path",
        required=True,
        type=Path,
        dest="test_data_path",
        help="Path to input test CSV",
)

    parser.add_argument(
        "-o",
        "--output-path",
        "--output_path",
        required=True,
        type=Path,
        dest="output_path",
        help="Path to output submission CSV",
    )
    parser.add_argument(
        "--models-path",
        type=Path,
        help="Models root; defaults to SHARED_MODELS_PATH or ./models locally",
    )
    parser.add_argument(
        "--pixel-preset",
        choices=tuple(PIXEL_PRESETS),
        default=DEFAULT_PIXEL_PRESET,
        help="Image resolution preset used by both training and inference",
    )
    parser.add_argument("--embed-batch-size", type=int)
    parser.add_argument("--llm-batch-size", type=int)

    args = parser.parse_args()

    for name in ("embed_batch_size", "llm_batch_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    # Heavy GPU dependencies are imported only for a real inference run. This keeps
    # `run.py --help` usable in the lightweight local experiment environment.
    from src.utils_embed_cuda import embed_data_cuda
    from src.utils_generate_cuda import generate_comments_cuda
    import torch

    # готовим пути
    data_path = args.test_data_path
    output_path = args.output_path
    images_path = data_path.parent / "images"
    models_dir = resolve_models_dir(args.models_path)
    model_embed_path = models_dir / EMBED_MODEL_NAME
    model_llm_path = models_dir / LLM_MODEL_NAME
    classifier_path = resolve_classifier_path()

    if not data_path.is_file():
        parser.error(f"Input CSV not found: {data_path}")

    if not images_path.is_dir():
        parser.error(f"Images directory not found: {images_path}")

    for model_path in (model_embed_path, model_llm_path):
        if not model_path.is_dir():
            parser.error(
                f"Model directory not found: {model_path}. Set SHARED_MODELS_PATH correctly."
            )
    if not classifier_path.is_file():
        parser.error(
            "Classifier artifact not found: expected "
            f"{CLASSIFIER_NPZ_PATH} or {CLASSIFIER_JOBLIB_PATH}"
        )

    has_cuda = torch.cuda.is_available()
    embed_batch_size = args.embed_batch_size or (
        DEFAULT_EMBED_BATCH_SIZE if has_cuda else DEFAULT_LOCAL_EMBED_BATCH_SIZE
    )
    llm_batch_size = args.llm_batch_size or (
        DEFAULT_LLM_BATCH_SIZE if has_cuda else DEFAULT_LOCAL_LLM_BATCH_SIZE
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_df = prepare_dataframe(data_path, images_path)

    # Step 2: extract embeddings for text+images (batch_size=128 for H100 80GB)
    current_embeddings = embed_data_cuda(
        str(model_embed_path), current_df,
        max_pixels=PIXEL_PRESETS[args.pixel_preset],
        batch_size=embed_batch_size,
    )

    # Step 3: load classification models and predict
    trained_logreg = ProductQualityPredictor.load(classifier_path)
    current_df['logreg_prob'], current_df['pred'] = trained_logreg.predict(
        current_embeddings, current_df['category']
    )

    # Step 4: generate comments
    comments = generate_comments_cuda(
        str(model_llm_path), current_df,
        batch_size=llm_batch_size,
    )

    # Step 5: patch comments and make them comply with length constraints
    current_df['result'] = format_results(comments, current_df['pred'].tolist())

    # Step 6: finalize
    result_df = current_df[['id', 'result']]
    result_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
