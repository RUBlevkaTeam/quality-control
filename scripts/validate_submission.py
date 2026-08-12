"""Validate an output CSV against the competition result format."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_RESULT_RE = re.compile(
    r"^<комментарий>(?P<comment>[\s\S]*)<вердикт>(?P<verdict>бан|не бан)$"
)


def validate_submission(input_path: Path, output_path: Path) -> None:
    source = pd.read_csv(input_path)
    result = pd.read_csv(output_path)
    if list(result.columns) != ["id", "result"]:
        raise ValueError(f"output columns must be exactly ['id', 'result'], got {list(result.columns)}")
    if result["id"].duplicated().any():
        raise ValueError("output contains duplicate id values")
    if len(result) != len(source) or set(result["id"]) != set(source["id"]):
        missing = set(source["id"]) - set(result["id"])
        extra = set(result["id"]) - set(source["id"])
        raise ValueError(
            f"output ids do not match input: rows={len(result)}/{len(source)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    errors: list[str] = []
    for row_number, value in enumerate(result["result"], start=2):
        match = _RESULT_RE.fullmatch(str(value))
        if not match:
            errors.append(f"row {row_number}: invalid result format")
            continue
        length = len(match.group("comment"))
        if not 50 <= length <= 300:
            errors.append(f"row {row_number}: comment length is {length}, expected 50..300")
        if len(errors) >= 10:
            break
    if errors:
        raise ValueError("submission validation failed:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original test CSV")
    parser.add_argument("--output", type=Path, required=True, help="Generated CSV")
    args = parser.parse_args()
    validate_submission(args.input, args.output)
    print(f"OK: {args.output} has valid ids, columns and result format")


if __name__ == "__main__":
    main()

