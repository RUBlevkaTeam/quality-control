#!/usr/bin/env python3
"""Build and validate the allowlisted baseline submission archive."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submit.zip"
ALLOWLIST = (
    "run.py",
    "metadata.json",
    "baseline_qwen3vl_bf16.joblib",
    "text_model.joblib",
    "image_retrieval.npz",
    "native/libhashmatch_linux_x86_64.so",
    "src/__init__.py",
    "src/constants.py",
    "src/image_retrieval.py",
    "src/output_validation.py",
    "src/rule_features.py",
    "src/text_model.py",
    "src/utils_data_prep.py",
    "src/utils_embed_cuda.py",
    "src/utils_generate_cuda.py",
    "src/utils_logreg.py",
    "src/utils_postprocess.py",
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
        raise ValueError(f"unexpected metadata: {metadata!r}")
    native = (ROOT / "native/libhashmatch_linux_x86_64.so").read_bytes()
    if not native.startswith(b"\x7fELF"):
        raise ValueError("native matcher is not a Linux ELF")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.image_retrieval import load_retrieval_index
    from src.text_model import TextQualityModel

    load_retrieval_index(ROOT / "image_retrieval.npz")
    TextQualityModel.load(ROOT / "text_model.joblib")

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in ALLOWLIST:
            info = zipfile.ZipInfo(relative, date_time=(2026, 8, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100755 << 16 if relative.endswith(".so") else 0o100644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes(), compresslevel=6)

    with zipfile.ZipFile(OUTPUT) as archive:
        if tuple(archive.namelist()) != ALLOWLIST:
            raise ValueError("ZIP contents differ from allowlist")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"{OUTPUT}\nsize={OUTPUT.stat().st_size}\nsha256={digest}\nfiles={len(ALLOWLIST)}")


if __name__ == "__main__":
    main()
