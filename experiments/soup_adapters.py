#!/usr/bin/env python3
"""Сид-суп: усреднение дельт W нескольких LoRA-адаптеров в один.

Усредняем дельты scale*(B@A) в fp32 (факторы A/B из разных прогонов живут в
разных подпространствах - пофакторное среднее бессмысленно). Среднее n дельт
ранга r имеет ранг до n*r, поэтому НЕ раскладываем обратно в ранг r (SVD-усечение
теряло ~37%), а конкатенируем блоки:

    mean_i scale*(B_i @ A_i) = B' @ A',  где
    A' = concat_rows(A_i),  B' = concat_cols((scale/n) * B_i)

и пишем адаптер с alpha=r' (scale'=1), чтобы LoRA-хук на инференсе применил
ровно усреднённую дельту, без потерь. Формат остаётся обычным адаптером.

    python3 soup_adapters.py --out qc_soup/adapter_all qc_seed*/adapter_fold0
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("adapters", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = [Path(a) for a in args.adapters]
    n = len(dirs)
    print(f"суп из {n} адаптеров: {[d.name for d in dirs]}")

    states, scales = [], []
    for d in dirs:
        cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
        states.append(load_file(str(d / "adapter_model.safetensors")))
        scales.append(float(cfg["lora_alpha"]) / float(cfg["r"]))

    a_keys = [k for k in states[0] if ".lora_A." in k]
    merged = {}
    total_r = 0
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        A_blocks, B_blocks = [], []
        for st, scale in zip(states, scales):
            A_blocks.append(st[a_key].to(torch.float32))
            B_blocks.append((scale / n) * st[b_key].to(torch.float32))
        A_cat = torch.cat(A_blocks, dim=0)   # (n*r, in)
        B_cat = torch.cat(B_blocks, dim=1)   # (out, n*r)
        merged[a_key] = A_cat.to(torch.bfloat16).contiguous()
        merged[b_key] = B_cat.to(torch.bfloat16).contiguous()
        total_r = A_cat.shape[0]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "adapter_model.safetensors"))
    cfg = json.loads((dirs[0] / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["r"] = total_r
    cfg["lora_alpha"] = total_r          # scale' = 1: дельта уже вложена в B'
    (out / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for extra in ("rules_version.txt",):
        if (dirs[0] / extra).is_file():
            shutil.copy(dirs[0] / extra, out / extra)
    print(f"сохранено: {out} (ранг {total_r}, {len(merged)//2} матриц)")


if __name__ == "__main__":
    main()
