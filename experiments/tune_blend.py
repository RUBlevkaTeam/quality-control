#!/usr/bin/env python3
"""Подбор весов ансамбля текст+VLM и порогов по family-OOF.

Вход (reports/):
  text_oof.csv   - из scripts/dump_text_oof.py (id, prob);
  oof_fold*.csv  - предсказания LoRA по фолдам (id, prob) из колаба;
  data.csv, families.csv - как обычно.

Перебирает w_text в сетке для каждой категории отдельно, порог - exact_best
на смешанных вероятностях, усредняет по фолдам. Пишет lora_blend.json и
печатает сравнение против чистой текстовой модели.

Локально считает секунды: только арифметика над готовыми вероятностями.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_model import exact_best_threshold


def main() -> None:
    reports = ROOT / "reports"
    text_oof = pd.read_csv(reports / "text_oof.csv")
    vlm_frames = [
        pd.read_csv(path)
        for path in sorted(reports.glob("oof_fold*.csv"))
    ]
    if not vlm_frames:
        raise SystemExit("нет oof_fold*.csv - сначала обучение в колабе")
    vlm_oof = (
        pd.concat(vlm_frames)
        .groupby("id")["prob"]
        .mean()
        .reset_index()
    )

    df = pd.read_csv(ROOT / "data.csv")[["id", "category", "label"]]
    corrections_path = ROOT / "label_corrections.csv"
    if corrections_path.exists():
        corrections = pd.read_csv(corrections_path)
        fix = dict(zip(corrections["id"], corrections["corrected_label"]))
        df.loc[df["id"].isin(fix), "label"] = df.loc[df["id"].isin(fix), "id"].map(fix)

    merged = (
        df.merge(text_oof, on="id", how="inner")
        .merge(vlm_oof.rename(columns={"prob": "vlm"}), on="id", how="left")
    )
    merged["vlm"] = merged["vlm"].fillna(merged["prob"])

    weights = {}
    thresholds = {}
    print(f"{'категория':<24}{'text F1':>9}{'best w_t':>10}{'ens F1':>9}{'порог':>8}")
    for category, group in merged.groupby("category"):
        y = group["label"].to_numpy(dtype=np.int64)
        p_text = group["prob"].to_numpy(dtype=np.float64)
        p_vlm = group["vlm"].to_numpy(dtype=np.float64)

        base_f1, _ = exact_best_threshold(p_text, y)
        best = None
        for w in np.linspace(1.0, 0.0, 21):
            f1, thr = exact_best_threshold(w * p_text + (1 - w) * p_vlm, y)
            if best is None or f1 > best[0]:
                best = (float(f1), float(w), float(thr))
        key = str(category).strip().lower()
        weights[key] = round(best[1], 2)
        thresholds[key] = round(best[2], 4)
        print(
            f"{str(category):<24}{base_f1:>9.4f}{best[1]:>10.2f}"
            f"{best[0]:>9.4f}{best[2]:>8.3f}"
        )

    mean_gain = None
    config = {"weights": weights, "thresholds": thresholds}
    out = ROOT / "lora_blend.json"
    out.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out}\n{json.dumps(config, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
