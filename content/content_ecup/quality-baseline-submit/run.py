import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from src.constants import PIXEL_PRESETS, DEFAULT_EMBED_BATCH_SIZE, DEFAULT_LLM_BATCH_SIZE
from src.utils_data_prep import prepare_dataframe
from src.utils_logreg import ProductQualityPredictor
from src.utils_postprocess import format_results
from src.utils_embed_cuda import embed_data_cuda
from src.utils_generate_cuda import generate_comments_cuda

# Classifier artifact path — Docker mounts artifacts at /app/artifacts/
CLASSIFIER_PATH = "baseline_qwen3vl_bf16.joblib"

# Models path: match evaluator's SHARED_MODELS_PATH convention
_SHARED_MODELS_DIR = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
MODEL_EMBED_PATH = os.path.join(_SHARED_MODELS_DIR, "Qwen/Qwen3-VL-Embedding-2B")
MODEL_LLM_PATH = os.path.join(_SHARED_MODELS_DIR, "Qwen/Qwen3.5-4B")


def main() -> None:
    parser = argparse.ArgumentParser(description="Product quality predictor submit pipeline")
    parser.add_argument("--test_data_path", type=str, help="test data path")
    parser.add_argument("--output_path", type=str, help="output file")
    args = parser.parse_args()

    # Step 1: read and prepare combined text + structured image paths
    data_path = Path(args.test_data_path)
    images_path = data_path.parent / "images"
    current_df = prepare_dataframe(data_path, images_path)

    # Step 2: extract embeddings for text+images (batch_size=128 for H100 80GB)
    current_embeddings = embed_data_cuda(
        MODEL_EMBED_PATH, current_df,
        max_pixels=PIXEL_PRESETS["M"],
        batch_size=DEFAULT_EMBED_BATCH_SIZE,
    )
    current_df['embedding'] = current_embeddings.tolist()

    # Step 3: load classification models and predict
    trained_logreg = ProductQualityPredictor.load(CLASSIFIER_PATH)
    current_df['logreg_prob'], current_df['pred'] = trained_logreg.predict(
        current_df['embedding'], current_df['category']
    )

    # Step 4: generate comments
    comments = generate_comments_cuda(
        MODEL_LLM_PATH, current_df,
        batch_size=DEFAULT_LLM_BATCH_SIZE,
    )

    # Step 5: patch comments and make them comply with length constraints
    current_df['result'] = format_results(comments, current_df['pred'].tolist())

    # Step 6: finalize
    result_df = current_df[['id', 'result']]
    result_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
