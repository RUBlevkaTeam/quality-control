# Must be set BEFORE torch is imported — prevents GPU memory fragmentation
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

from src.constants import PIXEL_PRESETS

# Qwen-specific util to resize image so that w*h <= max_pixels, keeping 28-pixel grid alignment
def _resize_image_to_max_pixels(img: Image.Image, max_pixels: int, resample=Image.LANCZOS) -> Image.Image:
    
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(28, (int(w * scale) // 28) * 28)
    new_h = max(28, (int(h * scale) // 28) * 28)
    return img.resize((new_w, new_h), resample)

# Compute attention-masked mean pooling over last_hidden_state
def _extract_pooled_emb(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    
    hidden = hidden_state
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    sum_emb = torch.sum(hidden * mask, dim=1)
    sum_m = torch.clamp(mask.sum(dim=1), min=1e-9)
    return (sum_emb / sum_m).squeeze(1).cpu().numpy().astype(np.float32)

# Embed a batch where ALL samples have at least one image
def _embed_batch_with_images(
    processor,
    model,
    texts: List[str],
    all_images: List[List[Image.Image]],
    max_pixels: int,
    device: str,
) -> np.ndarray:

    # Build text with vision tokens for each image
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

# Embed a batch with NO images — pure text fallback
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
   
    # Aggressively free memory before loading model
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model once
    _is_local = os.path.exists(embed_model_path) or (
        os.path.isabs(embed_model_path) and not embed_model_path.startswith(("http://", "https://", "file://"))
    )
    if _is_local:
        processor = AutoProcessor.from_pretrained(embed_model_path, local_files_only=True)
        model = AutoModel.from_pretrained(
            embed_model_path, torch_dtype=torch.float16, local_files_only=True, trust_remote_code=True
        )
    else:
        processor = AutoProcessor.from_pretrained(embed_model_path)
        model = AutoModel.from_pretrained(
            embed_model_path, torch_dtype=torch.float16, trust_remote_code=True
        )
    model = model.to(device).eval()

    n = len(df)
    all_embeddings = np.zeros((n, 0), dtype=np.float32)

    # Process in batches — true parallel inference
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_df = df.iloc[start:end].reset_index(drop=True)

        # Determine which samples have images and which don't
        has_images_mask = batch_df['image_paths'].apply(
            lambda x: bool(x and len(x) > 0)
        )
        num_with_images = has_images_mask.sum()

        # Separate into two groups
        df_with_imgs = batch_df[has_images_mask].reset_index(drop=True)
        df_text_only = batch_df[~has_images_mask].reset_index(drop=True)

        # Compute embeddings for each group
        emb_with_list: List[np.ndarray] = []
        emb_text_list: List[np.ndarray] = []

        # Group A: embed samples WITH images (true batch, fallback per-sample on OOM)
        if num_with_images > 0:
            texts_with_imgs = df_with_imgs['text'].tolist()
            all_images = []
            for _, row in df_with_imgs.iterrows():
                imgs = []
                for p in row['image_paths']:
                    raw = Image.open(p).convert("RGB")
                    imgs.append(_resize_image_to_max_pixels(raw, max_pixels))
                all_images.append(imgs)

            try:
                emb_with = _embed_batch_with_images(
                    processor, model, texts_with_imgs, all_images, max_pixels, device
                )
                emb_with_list = list(emb_with)
            except torch.cuda.OutOfMemoryError:
                # Fallback: embed each sample individually
                emb_with_list = []
                for idx in range(len(df_with_imgs)):
                    single_imgs = all_images[idx]
                    try:
                        single_emb = _embed_batch_with_images(
                            processor, model,
                            [texts_with_imgs[idx]],
                            [single_imgs],
                            max_pixels, device,
                        )
                        emb_with_list.append(single_emb[0])
                    except torch.cuda.OutOfMemoryError:
                        # Even single sample OOMs — try without images
                        for img in single_imgs:
                            img.close()
                        single_emb = _embed_batch_text_only(
                            processor, model,
                            [texts_with_imgs[idx]],
                            device,
                        )
                        emb_with_list.append(single_emb[0])
                    finally:
                        for img in single_imgs:
                            try:
                                img.close()
                            except Exception:
                                pass

        # Group B: embed samples WITHOUT images (text-only batch)
        if len(df_text_only) > 0:
            texts_only = df_text_only['text'].tolist()
            emb_text = _embed_batch_text_only(processor, model, texts_only, device)
            emb_text_list = list(emb_text)

        # Assemble batch_result using list of 1-D arrays, then vstack at end
        # Place in correct order
        emb_with_iter = iter(emb_with_list)
        emb_text_iter = iter(emb_text_list)
        batch_result_rows = []
        for i in range(len(batch_df)):
            if has_images_mask.values[i]:
                batch_result_rows.append(next(emb_with_iter))
            else:
                batch_result_rows.append(next(emb_text_iter))

        # Stack rows into 2-D array
        batch_result = np.stack(batch_result_rows)

        # Accumulate results
        if all_embeddings.shape[1] == 0:
            all_embeddings = batch_result.copy()
        else:
            all_embeddings = np.concatenate([all_embeddings, batch_result], axis=0)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return all_embeddings
