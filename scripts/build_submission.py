"""Build the minimal ZIP archive expected by the competition evaluator."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_ROOT_FILES = ("run.py", "metadata.json")
CLASSIFIER_FILES = (
    "baseline_qwen3vl_bf16.npz",
    "baseline_qwen3vl_bf16.joblib",
)
SOURCE_FILES = (
    "__init__.py",
    "constants.py",
    "utils_data_prep.py",
    "utils_embed_cuda.py",
    "utils_generate_cuda.py",
    "utils_logreg.py",
    "utils_postprocess.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "dist/quality-control-submission.zip",
    )
    args = parser.parse_args()

    classifier_file = next(
        (name for name in CLASSIFIER_FILES if (PROJECT_DIR / name).is_file()),
        None,
    )
    if classifier_file is None:
        raise FileNotFoundError(
            f"required classifier is missing; expected one of {CLASSIFIER_FILES}"
        )
    root_files = (*BASE_ROOT_FILES, classifier_file)
    for relative in root_files:
        path = PROJECT_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"required submission file is missing: {path}")
    metadata = json.loads((PROJECT_DIR / "metadata.json").read_text(encoding="utf-8"))
    if set(metadata) != {"image", "entry_point"}:
        raise ValueError("metadata.json must contain exactly image and entry_point")

    source_files = [PROJECT_DIR / "src" / name for name in SOURCE_FILES]
    missing_sources = [str(path) for path in source_files if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"submission modules are missing: {missing_sources}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in root_files:
            archive.write(PROJECT_DIR / relative, arcname=relative)
        for path in source_files:
            archive.write(path, arcname=f"src/{path.name}")

    with zipfile.ZipFile(args.output) as archive:
        bad_file = archive.testzip()
        names = set(archive.namelist())
    if bad_file:
        raise RuntimeError(f"corrupt file in archive: {bad_file}")
    required_names = set(root_files) | {"src/__init__.py"}
    if not required_names.issubset(names):
        raise RuntimeError(f"archive is missing: {sorted(required_names - names)}")
    size_mb = args.output.stat().st_size / 1024**2
    print(f"Built {args.output} ({size_mb:.2f} MiB, {len(names)} files)")


if __name__ == "__main__":
    main()
