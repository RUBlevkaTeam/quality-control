# выполняем до торча, чтобы не допустить фрагментации VRAM
import os as _os
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from pathlib import Path

from src.constants import PIXEL_PRESETS

# скейлинг под максмимальное количество пикселей
def _resize_image_to_max_pixels(img: Image.Image, max_pixels: int, resample=Image.LANCZOS) -> Image.Image:
    
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(28, (int(w * scale) // 28) * 28)
    new_h = max(28, (int(h * scale) // 28) * 28)
    return img.resize((new_w, new_h), resample)

# применяем mean_pooling к последнему слою
def _extract_pooled_emb(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    
    hidden = hidden_state
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    sum_emb = torch.sum(hidden * mask, dim=1)
    sum_m = torch.clamp(mask.sum(dim=1), min=1e-9)
    return (sum_emb / sum_m).cpu().numpy().astype(np.float32)

# создаем эмбеддинги примерам с фотографиями
def _embed_batch_with_images(
    processor,
    model,
    texts: List[str],
    all_images: List[List[Image.Image]],
    device: str) -> np.ndarray:

    # строим специальные визуал токены
    vision_start = getattr(processor, "vision_start_token", "<|vision_start|>")
    vision_end = getattr(processor, "vision_end_token", "<|vision_end|>")
    image_tok = getattr(processor, "image_token", "<|image|>")
    vision_placeholder = vision_start + image_tok + vision_end

    processed_texts = []
    for text, imgs in zip(texts, all_images):
        placeholder_str = vision_placeholder * len(imgs)
        processed_texts.append(text + placeholder_str)

    inputs = processor(
        text=processed_texts,
        images=all_images,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return _extract_pooled_emb(outputs.last_hidden_state, inputs["attention_mask"])

# обработка примеров без фото
def _embed_batch_text_only(
    processor,
    model,
    texts: List[str],
    device: str,
) -> np.ndarray:
    
    inputs = processor(text=texts, images=None, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return _extract_pooled_emb(outputs.last_hidden_state, inputs["attention_mask"])


# Load embedding model and process dataframe with batch inference
# Falls back to per-sample inference for a batch if it OOMs
def embed_data_cuda(
    embed_model_path: str,
    df: pd.DataFrame,
    max_pixels: int = 128 * 28 * 28,
    batch_size: int = 128,
) -> np.ndarray:
   
    def _resolve_device_and_dtype():
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.bfloat16

        if torch.backends.mps.is_available():
            return torch.device("mps"), torch.float16

        return torch.device("cpu"), torch.float32

    device, dtype = _resolve_device_and_dtype()

    # Load model once
    model_path = Path(embed_model_path)

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Embedding model not found: {model_path}"
        )

    processor = AutoProcessor.from_pretrained(embed_model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        embed_model_path, torch_dtype=dtype, local_files_only=True, trust_remote_code=True
    )
    model = model.to(device).eval()

    n = len(df)
    all_embeddings = []
    # Process in batches — true parallel inference
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_df = df.iloc[start:end].reset_index(drop=True)

        # определяем группы с изображениями и без
        has_images_mask = batch_df['image_paths'].map(bool)
        num_with_images = has_images_mask.sum()

        # разделяем на 2 группы
        df_with_imgs = batch_df[has_images_mask].reset_index(drop=True)
        df_text_only = batch_df[~has_images_mask].reset_index(drop=True)

        # создаем пустые эмбеддинги
        emb_with_list: List[np.ndarray] = []
        emb_text_list: List[np.ndarray] = []

        # обрабатываем с изображениями
        if num_with_images > 0:
            texts_with_imgs = df_with_imgs['text'].tolist()
            all_images = []
            try:
                for _, row in df_with_imgs.iterrows():
                    imgs = []
                    all_images.append(imgs)
                    for p in row['image_paths']:
                        with Image.open(p) as source:
                            rgb_image = source.convert("RGB")

                        resized_image = _resize_image_to_max_pixels(rgb_image, max_pixels)
                        if resized_image is not rgb_image:
                            rgb_image.close()
                        imgs.append(resized_image)

                try:
                    emb_with = _embed_batch_with_images(
                        processor, model, texts_with_imgs, all_images, device
                    )
                    emb_with_list = list(emb_with)
                except torch.cuda.OutOfMemoryError:
                    # Обработка индивидуально при исключении по ООМ
                    torch.cuda.empty_cache()
                    emb_with_list = []
                    for idx in range(len(df_with_imgs)):
                        single_imgs = all_images[idx]
                        try:
                            single_emb = _embed_batch_with_images(
                                processor, model,
                                [texts_with_imgs[idx]],
                                [single_imgs],
                                device,
                            )
                            emb_with_list.append(single_emb[0])
                        except torch.cuda.OutOfMemoryError:
                            # Обработка 2 исключения по ООМ без фото
                            torch.cuda.empty_cache()
                            single_emb = _embed_batch_text_only(
                                processor, model,
                                [texts_with_imgs[idx]],
                                device,
                            )
                            emb_with_list.append(single_emb[0])
            finally:
                for product_images in all_images:
                    for image in product_images:
                        image.close()

        # обрабатываем без изображений
        if len(df_text_only) > 0:
            texts_only = df_text_only['text'].tolist()
            emb_text = _embed_batch_text_only(processor, model, texts_only, device)
            emb_text_list = list(emb_text)

        # итерируемся и восстанавливаем правильный порядок
        emb_with_iter = iter(emb_with_list)
        emb_text_iter = iter(emb_text_list)
        batch_result_rows = []
        for i in range(len(batch_df)):
            if has_images_mask.values[i]:
                batch_result_rows.append(next(emb_with_iter))
            else:
                batch_result_rows.append(next(emb_text_iter))

        # создаем матрицу из векторов
        batch_result = np.stack(batch_result_rows)

        # накапливаем результаты
        if len(all_embeddings) == 0:
            all_embeddings = batch_result.copy()
        else:
            all_embeddings.append(batch_result)

    all_embeddings = np.vstack(all_embeddings)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return all_embeddings
