import html
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_TEXT_COLUMNS = ("name", "category", "description")

# имя тега только латиницей: иначе после раскрытия сущностей конструкции
# вроде <складной> будут вырезаны как теги
_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,19}(?:\s[^>]{0,300})?/?>")
_WS_RE = re.compile(r"\s+")

# убирает HTML разметку
def _clean(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").map(html.unescape).map(html.unescape)
    # тег заменяем ПРОБЕЛОМ, иначе <p>не</p><p>является</p> склеится в «неявляется»
    cleaned = cleaned.str.replace(_TAG_RE, " ", regex=True)
    return cleaned.str.replace(_WS_RE, " ", regex=True).str.strip()

# склеивает колонки одной строки в один текст
def _build_text_vectorized(df: pd.DataFrame) -> pd.Series:
    return (
        "Название: " + _clean(df["name"]) + "\n"
        "Категория: " + _clean(df["category"]) + "\n"
        "Описание: " + _clean(df["description"])
    )

# проходит папки с картинками и отсекает не нужное
def _scan_images_root(images_path: Path) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    try:
        entries = list(os.scandir(images_path))
    except OSError:
        return index  # папки нет значит берем только текст

    for entry in entries:
        try:
            if not entry.is_dir():  # отсекает .DS_Store в корне
                continue
            files = [
                inner.path
                for inner in os.scandir(entry.path)
                if inner.is_file()  # директория "photo.jpg" - не изображение
                and os.path.splitext(inner.name)[1].lower() in _VALID_EXTENSIONS
            ]
        except OSError:
            continue  # одна битая папка не роняет ничего
        if files:
            index[entry.name] = sorted(files)
    return index

# сопоставляет айди товара и айди папки с картинками и выдает список файлов картинок для каждого товара
def _find_images_vectorized(df: pd.DataFrame, images_path: Path) -> pd.Series:
    index = _scan_images_root(images_path)
    # astype(str) обязателен: id - int64, а ключи словаря - имена папок,
    # без приведения ничего не найдётся и картинки потеряются молча
    return df["id"].astype(str).map(lambda key: index.get(key, []))

# читает CSV и добавляет text, rule_text, image_paths, n_images
def prepare_dataframe(data_path: str | Path, images_path: str | Path) -> pd.DataFrame:
    current_df = pd.read_csv(data_path)

    # data.csv сохранён без index = False и тащит лишний столбец
    junk = [c for c in current_df.columns if str(c).startswith("Unnamed:")]
    if junk:
        current_df = current_df.drop(columns=junk)

    # без id невозможно сопоставить ответы товарам
    if "id" not in current_df.columns:
        raise ValueError(f"нет колонки 'id', есть: {list(current_df.columns)}")

    # в тестовом наборе поле может отсутствовать - пишем туда пустоту ""
    for column in _TEXT_COLUMNS:
        if column not in current_df.columns:
            current_df[column] = ""

    current_df["text"] = _build_text_vectorized(current_df)
    current_df["rule_text"] = _build_rule_text_vectorized(current_df)
    current_df["image_paths"] = _find_images_vectorized(current_df, Path(images_path))
    current_df["n_images"] = current_df["image_paths"].map(len)

    return current_df