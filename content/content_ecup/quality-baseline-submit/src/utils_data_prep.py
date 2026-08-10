"""Optimized dataframe preparation with vectorized string operations."""

import pandas as pd
from pathlib import Path
from typing import List

# Valid image extensions
_VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


# Vectorized text construction
def _build_text_vectorized(df: pd.DataFrame) -> pd.Series:

    return (
        'Название: ' + df['name'].astype(str) + '\n'
        'Категория: ' + df['category'].astype(str) + '\n'
        'Описание: ' + df['description'].astype(str)
    )

# Find valid image files for a single product id
def _find_images_for_id(id_val, images_path: Path) -> List[str]:
    
    img_dir = images_path / str(id_val)
    if not img_dir.exists():
        return []
    return [str(img_dir / f) for f in sorted(img_dir.iterdir())
            if Path(f).suffix.lower() in _VALID_EXTENSIONS]

# Vectorized image path discovery based on  id column
def _find_images_vectorized(df: pd.DataFrame, images_path: Path) -> pd.Series:
    
    return df['id'].apply(_find_images_for_id, images_path=images_path)

# Load CSV and prepare text + image_paths columns
def prepare_dataframe(data_path: str, images_path: str) -> pd.DataFrame:
    
    current_df = pd.read_csv(data_path)
    img_dir = Path(images_path)

    # Vectorized text construction
    current_df['text'] = _build_text_vectorized(current_df)

    # Image path discovery — must remain row-wise due to variable file counts
    current_df['image_paths'] = _find_images_vectorized(current_df, img_dir)

    return current_df