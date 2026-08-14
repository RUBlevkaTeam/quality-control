#!/usr/bin/env python3
"""Build local and evaluator-native Hamming kernels with no dependencies."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/hashmatch.cpp"


def _compiler() -> str:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("clang++/g++ was not found")
    return compiler


def build_local() -> Path:
    compiler = _compiler()
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        output = ROOT / f"native/libhashmatch_darwin_{machine}.dylib"
        command = [compiler, "-O3", "-std=c++17", "-dynamiclib", str(SOURCE), "-o", str(output)]
    elif system == "linux":
        output = ROOT / f"native/libhashmatch_linux_{machine}.so"
        command = [compiler, "-O3", "-std=c++17", "-shared", "-fPIC", str(SOURCE), "-o", str(output)]
    else:
        raise RuntimeError(f"unsupported local platform: {system}/{machine}")
    subprocess.run(command, check=True)
    return output


def build_linux_x86_64() -> Path:
    zig = shutil.which("zig")
    if zig is None:
        try:
            import ziglang

            candidate = Path(ziglang.__file__).resolve().parent / "zig"
            zig = str(candidate) if candidate.is_file() else None
        except ImportError:
            zig = None
    if zig is None:
        raise RuntimeError(
            "cross-building the evaluator ELF requires zig "
            "(development-only: python -m pip install ziglang)"
        )
    output = ROOT / "native/libhashmatch_linux_x86_64.so"
    command = [
        zig,
        "c++",
        "-target",
        "x86_64-linux-gnu",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fno-exceptions",
        "-fno-rtti",
        "-nostdlib",
        "-Wl,--no-undefined",
        str(SOURCE),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="build for the current host")
    parser.add_argument("--linux-x86-64", action="store_true", help="cross-build evaluator ELF")
    args = parser.parse_args()
    if not args.local and not args.linux_x86_64:
        args.local = True

    outputs = []
    if args.local:
        outputs.append(build_local())
    if args.linux_x86_64:
        outputs.append(build_linux_x86_64())
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
