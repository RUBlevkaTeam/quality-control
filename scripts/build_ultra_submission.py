#!/usr/bin/env python3
"""Build a deterministic, allowlisted ultra submission ZIP."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist/ecup_quality_ultra.zip"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWLIST = (
    "metadata.json",
    "run.py",
    "baseline_qwen3vl_bf16.joblib",
    "ultra_quality.joblib",
    "src/__init__.py",
    "src/constants.py",
    "src/output_validation.py",
    "src/ultra_explain.py",
    "src/ultra_features.py",
    "src/ultra_model.py",
    "src/ultra_predictor.py",
    "src/ultra_retrieval.py",
    "src/utils_data_prep.py",
    "src/utils_embed_cuda.py",
    "src/utils_generate_cuda.py",
    "src/utils_logreg.py",
)


def main() -> None:
    missing = [name for name in ALLOWLIST if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"submission files are missing: {missing}")

    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    if metadata != {
        "image": "odsai/ecup26-quality-baseline:1.0",
        "entry_point": "python -u run.py",
    }:
        raise ValueError(f"unexpected metadata.json: {metadata!r}")
    from src.ultra_predictor import load_ultra_artifact, predict_fast

    artifact = load_ultra_artifact(ROOT / "ultra_quality.joblib")
    smoke = pd.DataFrame(
        {
            "id": ["schema-bad", "schema-fire"],
            "name": ["Биологически активная добавка", "Газовая горелка"],
            "description": ["Добавка к пище", "Газовый баллон не входит в комплект"],
            "category": ["БАД", "Легковоспламеняющиеся"],
            "image_paths": [[], []],
        }
    )
    smoke_result = predict_fast(artifact, smoke)
    if len(smoke_result["predictions"]) != len(smoke):
        raise ValueError("ultra artifact smoke prediction returned the wrong row count")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for relative in ALLOWLIST:
            source = ROOT / relative
            info = zipfile.ZipInfo(relative, date_time=(2026, 8, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compresslevel=6)

    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if tuple(names) != ALLOWLIST:
            raise ValueError(f"ZIP contents differ from allowlist: {names!r}")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe ZIP path: {name}")

    size = OUTPUT.stat().st_size
    if size >= 5_000_000_000:
        raise ValueError(f"submission exceeds 5 GB: {size}")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"{OUTPUT}\nsize={size}\nsha256={digest}\nfiles={len(ALLOWLIST)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
