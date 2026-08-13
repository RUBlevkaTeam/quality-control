"""Read and prepare product data for training and inference."""

import html
import re
from pathlib import Path

import pandas as pd

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9a-zа-я]+")


def clean_text(series: pd.Series) -> pd.Series:
    """Remove HTML and normalize whitespace in a text column."""
    return (
        series.fillna("")
        .astype(str)
        .map(html.unescape)
        .map(html.unescape)
        .str.replace(_HTML_TAG, " ", regex=True)
        .str.replace(_WHITESPACE, " ", regex=True)
        .str.strip()
    )


def build_text(df: pd.DataFrame) -> pd.Series:
    """Build the text passed to the Qwen embedding model."""
    return (
        "Название: " + clean_text(df["name"]) + "\n"
        "Категория: " + clean_text(df["category"]) + "\n"
        "Описание: " + clean_text(df["description"])
    )


def build_group_key(df: pd.DataFrame) -> pd.Series:
    """Build a normalized key used to keep duplicate cards in one fold."""
    def normalize(series: pd.Series) -> pd.Series:
        return (
            clean_text(series)
            .str.lower()
            .str.replace("ё", "е", regex=False)
            .str.replace(_NON_ALNUM, "", regex=True)
        )

    return normalize(df["name"]) + "\n" + normalize(df["description"])


def find_images(product_id, images_path: Path) -> list[str]:
    product_dir = images_path / str(product_id)
    if not product_dir.is_dir():
        return []
    return [
        str(path)
        for path in sorted(product_dir.iterdir())
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ]


def read_dataframe(data_path: str | Path) -> pd.DataFrame:
    """Read a competition CSV and remove its accidental index column."""
    df = pd.read_csv(data_path)
    junk_columns = [column for column in df if str(column).startswith("Unnamed:")]
    return df.drop(columns=junk_columns)


def prepare_dataframe(data_path: str | Path, images_path: str | Path) -> pd.DataFrame:
    """Read data and add cleaned Qwen text and image paths."""
    df = read_dataframe(data_path)
    missing = {"id", "category"} - set(df.columns)
    if missing:
        raise ValueError(f"input CSV is missing required columns: {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError("input CSV must contain unique id values")

    for column in ("name", "description"):
        if column not in df:
            df[column] = ""

    df["text"] = build_text(df)
    df["image_paths"] = df["id"].map(
        lambda product_id: find_images(product_id, Path(images_path))
    )

    return df
