"""Ultra E-CUP quality-control submission entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# The evaluator is offline.  Set these before importing Transformers anywhere.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from src.ultra_explain import build_comment_variants, choose_comment_variant, format_result
from src.ultra_predictor import (
    apply_vlm_tiebreak,
    load_ultra_artifact,
    predict_fast,
    select_vlm_candidates,
)
from src.utils_data_prep import prepare_dataframe
from src.output_validation import validate_csv_file


ROOT = Path(__file__).resolve().parent
ULTRA_ARTIFACT_PATH = ROOT / "ultra_quality.joblib"
BASELINE_CLASSIFIER_PATH = ROOT / "baseline_qwen3vl_bf16.joblib"

_SHARED_MODELS_DIR = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
EMBED_MODEL_PATH = _SHARED_MODELS_DIR / "Qwen/Qwen3-VL-Embedding-2B"
LLM_MODEL_PATH = _SHARED_MODELS_DIR / "Qwen/Qwen3.5-4B"


def _validate_input(dataframe: pd.DataFrame) -> None:
    required = {"id", "name", "description", "category"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"input CSV misses columns: {missing}")
    if dataframe["id"].isna().any():
        raise ValueError("input contains an empty id")
    if dataframe["id"].astype(str).duplicated().any():
        raise ValueError("input ids must be unique")
    if dataframe["category"].isna().any():
        raise ValueError("input contains an empty category")
    allowed = {"БАД", "Легковоспламеняющиеся"}
    unexpected = sorted(set(dataframe["category"].astype(str)) - allowed)
    if unexpected:
        raise ValueError(f"unsupported categories: {unexpected}")


def _run_vlm_tiebreak(dataframe: pd.DataFrame, result: dict[str, object]) -> int:
    candidates = select_vlm_candidates(dataframe, result)
    if not candidates:
        return 0

    # Heavy dependencies stay lazy so CLI validation and CPU tests do not need
    # a GPU environment.  A VLM failure never prevents a valid submission.
    try:
        from src.constants import PIXEL_PRESETS
        from src.utils_embed_cuda import embed_data_cuda
        from src.utils_logreg import ProductQualityPredictor

        subset = dataframe.iloc[candidates].reset_index(drop=True)
        embeddings = embed_data_cuda(
            str(EMBED_MODEL_PATH),
            subset,
            max_pixels=PIXEL_PRESETS["M"],
            batch_size=min(128, max(1, len(subset))),
        )
        baseline = ProductQualityPredictor.load(str(BASELINE_CLASSIFIER_PATH))
        probabilities, _ = baseline.predict(embeddings, subset["category"].tolist())
        apply_vlm_tiebreak(dataframe, result, candidates, probabilities)
        return len(candidates)
    except Exception as exc:  # noqa: BLE001 - valid output is the fail-safe
        print(f"[ultra] VLM tie-break skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0


def _build_comment_variants(
    dataframe: pd.DataFrame,
    result: dict[str, object],
) -> list[tuple[str, str]]:
    variants = []
    predictions = np.asarray(result["predictions"], dtype=np.int8)
    for position, (_, row) in enumerate(dataframe.iterrows()):
        variants.append(
            build_comment_variants(
                category=row["category"],
                prediction=int(predictions[position]),
                flags=result["flags"][position],
                reason=result["reasons"][position],
            )
        )
    return variants


def _rewrite_comments_with_llm(
    dataframe: pd.DataFrame,
    result: dict[str, object],
    comment_variants: list[tuple[str, str]],
) -> tuple[list[str], bool]:
    try:
        from src.utils_generate_cuda import generate_grounded_comments_cuda

        predictions = np.asarray(result["predictions"], dtype=np.int8)
        generated_choices = generate_grounded_comments_cuda(
            str(LLM_MODEL_PATH),
            dataframe,
            predictions.tolist(),
            comment_variants,
            batch_size=64,
            max_new_tokens=4,
        )
        if len(generated_choices) != len(comment_variants):
            raise ValueError("LLM returned the wrong number of comment choices")
        comments = [
            choose_comment_variant(choice, variants)
            for choice, variants in zip(generated_choices, comment_variants)
        ]
        return comments, True
    except Exception as exc:  # noqa: BLE001 - grounded templates are the fail-safe
        print(f"[ultra] LLM comments skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return [variants[0] for variants in comment_variants], False


def _build_output(
    dataframe: pd.DataFrame,
    result: dict[str, object],
    comments: list[str],
) -> pd.DataFrame:
    predictions = np.asarray(result["predictions"], dtype=np.int8)
    formatted = [
        format_result(comment, int(prediction))
        for comment, prediction in zip(comments, predictions)
    ]
    return pd.DataFrame({"id": dataframe["id"].tolist(), "result": formatted})


def main() -> None:
    parser = argparse.ArgumentParser(description="E-CUP ultra product quality predictor")
    parser.add_argument(
        "--test_data_path",
        "--test-data-path",
        "-i",
        dest="test_data_path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        "-o",
        dest="output_path",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.test_data_path)
    _validate_input(raw)
    dataframe = prepare_dataframe(args.test_data_path, args.test_data_path.parent / "images")

    artifact = load_ultra_artifact(ULTRA_ARTIFACT_PATH)
    result = predict_fast(artifact, dataframe)
    routed = _run_vlm_tiebreak(dataframe, result)
    comment_variants = _build_comment_variants(dataframe, result)
    comments, llm_used = _rewrite_comments_with_llm(dataframe, result, comment_variants)
    output = _build_output(dataframe, result, comments)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    validate_csv_file(args.output_path, dataframe["id"].tolist())

    positives = int(np.asarray(result["predictions"], dtype=np.int8).sum())
    print(
        f"[ultra] rows={len(dataframe)} positives={positives} vlm_routed={routed} "
        f"llm_comments={int(llm_used)} "
        f"output={args.output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
