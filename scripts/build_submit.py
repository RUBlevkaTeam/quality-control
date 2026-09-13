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
    "template_transfer.json",
    "native/libhashmatch_linux_x86_64.so",
    "src/__init__.py",
    "src/constants.py",
    "src/device.py",
    "src/e5_channel.py",
    "src/image_retrieval.py",
    "src/output_validation.py",
    "src/retrieval_memory.py",
    "src/rule_features.py",
    "src/template_transfer.py",
    "src/text_model.py",
    "src/utils_data_prep.py",
    "src/utils_embed_cuda.py",
    "src/utils_generate_cuda.py",
    "src/comment_fallback.py",
    "src/brand_channel.py",
    "src/knn_transfer.py",
    "src/utils_logreg.py",
    "src/utils_postprocess.py",
    "src/vlm_ocr.py",
    "src/lora_classifier.py",
)

# Опциональные артефакты LoRA-ансамбля: попадают в архив только если собраны.
OPTIONAL = (
    "qc_lora_adapter/adapter_model.safetensors",
    "qc_lora_adapter/adapter_config.json",
    # маркер версии промпта правил: без него инференс молча откатится на v1
    # при v2-адаптере (train/inference-рассинхрон)
    "qc_lora_adapter/rules_version.txt",
    "lora_blend.json",
    "brand_stats.json",
    "knn_transfer.npz",
    "knn_transfer.json",
    # stub_comments.flag сюда НЕ включать: забытый диагностический флаг
    # уехал бы в боевой архив и заменил все комментарии заглушками
)


# Забытый в списке модуль не ломает сборку, а тихо роняет text_model в
# контейнере на ImportError - и прогон уходит по запасному пути с 0.505.
# Поэтому сверяем список с фактическими импортами, а не глазами.
def _imported_src_modules() -> set[str]:
    import ast

    found: set[str] = set()
    for name in ALLOWLIST:
        if not name.endswith(".py"):
            continue
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                found.add((node.module or "").split(".", 1)[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src."):
                        found.add(alias.name.split(".", 1)[1])
    return found


def main() -> None:
    optional = [name for name in OPTIONAL if (ROOT / name).is_file()]
    allowlist = tuple(ALLOWLIST) + tuple(optional)
    missing = [name for name in ALLOWLIST if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"submission files are missing: {missing}")
    packed = {name[4:-3] for name in ALLOWLIST if name.startswith("src/")}
    absent = sorted(m for m in _imported_src_modules() - packed if (ROOT / f"src/{m}.py").is_file())
    if absent:
        raise FileNotFoundError(f"imported but not packed: {absent}")
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
        for relative in allowlist:
            info = zipfile.ZipInfo(relative, date_time=(2026, 8, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100755 << 16 if relative.endswith(".so") else 0o100644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes(), compresslevel=6)

    with zipfile.ZipFile(OUTPUT) as archive:
        if tuple(archive.namelist()) != allowlist:
            raise ValueError("ZIP contents differ from allowlist")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    extras = ", ".join(optional) if optional else "нет"
    print(f"{OUTPUT}\nsize={OUTPUT.stat().st_size}\nsha256={digest}\nfiles={len(allowlist)}\nlora-артефакты: {extras}")


if __name__ == "__main__":
    main()
