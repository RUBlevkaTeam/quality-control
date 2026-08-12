"""Run the locally trained text baseline and create a valid submission-shaped CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.metrics import apply_thresholds
from src.text_baseline import clean_text
from src.utils_postprocess import format_results

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "artifacts/text_baseline.joblib",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input CSV not found: {args.input}")
    if not args.model.is_file():
        parser.error(f"model artifact not found: {args.model}; train it first")
    df = pd.read_csv(args.input)
    predictor = joblib.load(args.model)
    probabilities = predictor.predict_proba(df)
    predictions = apply_thresholds(
        probabilities, df["category"].astype(str), predictor.thresholds
    )

    names = clean_text(df["name"] if "name" in df else pd.Series("", index=df.index))
    comments = [
        (
            f"Карточка товара «{name[:80]}» по текстовым признакам соответствует "
            f"правилам категории {category}."
        )
        for name, category in zip(names, df["category"].astype(str))
    ]
    results = format_results(comments, predictions.tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "result": results}).to_csv(args.output, index=False)
    print(f"Saved {len(df):,} predictions to {args.output}")


if __name__ == "__main__":
    main()

