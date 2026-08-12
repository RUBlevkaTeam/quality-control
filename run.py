import argparse
import os
from pathlib import Path

from src.constants import PIXEL_PRESETS, DEFAULT_EMBED_BATCH_SIZE, DEFAULT_LLM_BATCH_SIZE
from src.utils_data_prep import prepare_dataframe
from src.utils_logreg import ProductQualityPredictor
from src.utils_postprocess import format_results

PROJECT_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = PROJECT_DIR / "baseline_qwen3vl_bf16.joblib" # директория классификатора

# Models path: match evaluator's SHARED_MODELS_PATH convention
_SHARED_MODELS_DIR = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
MODEL_EMBED_PATH = _SHARED_MODELS_DIR / "Qwen/Qwen3-VL-Embedding-2B"
MODEL_LLM_PATH = _SHARED_MODELS_DIR / "Qwen/Qwen3.5-4B"

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

    args = parser.parse_args()

    # Heavy GPU dependencies are imported only for a real inference run. This keeps
    # `run.py --help` usable in the lightweight local experiment environment.
    from src.utils_embed_cuda import embed_data_cuda
    from src.utils_generate_cuda import generate_comments_cuda

    # готовим пути
    data_path = args.test_data_path
    output_path = args.output_path
    images_path = data_path.parent / "images"

    if not data_path.is_file():
        parser.error(f"Input CSV not found: {data_path}")

    if not images_path.is_dir():
        parser.error(f"Images directory not found: {images_path}")

    for model_path in (MODEL_EMBED_PATH, MODEL_LLM_PATH):
        if not model_path.is_dir():
            parser.error(
                f"Model directory not found: {model_path}. Set SHARED_MODELS_PATH correctly."
            )
    if not CLASSIFIER_PATH.is_file():
        parser.error(f"Classifier artifact not found: {CLASSIFIER_PATH}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_df = prepare_dataframe(data_path, images_path)

    # Step 2: extract embeddings for text+images (batch_size=128 for H100 80GB)
    current_embeddings = embed_data_cuda(
        str(MODEL_EMBED_PATH), current_df,
        max_pixels=PIXEL_PRESETS["M"],
        batch_size=DEFAULT_EMBED_BATCH_SIZE,
    )

    # Step 3: load classification models and predict
    trained_logreg = ProductQualityPredictor.load(CLASSIFIER_PATH)
    current_df['logreg_prob'], current_df['pred'] = trained_logreg.predict(
        current_embeddings, current_df['category']
    )

    # Step 4: generate comments
    comments = generate_comments_cuda(
        str(MODEL_LLM_PATH), current_df,
        batch_size=DEFAULT_LLM_BATCH_SIZE,
    )

    # Step 5: patch comments and make them comply with length constraints
    current_df['result'] = format_results(comments, current_df['pred'].tolist())

    # Step 6: finalize
    result_df = current_df[['id', 'result']]
    result_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
