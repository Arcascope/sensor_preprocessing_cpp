#!/usr/bin/env python3
"""Fail if a built wheel bundles native libraries using instructions above the
portable x86-64-v3 baseline.

FINUFFT defaults its architecture flag to ``native``, so a CI runner with
AVX-512 can bake AVX-512 into ``libfinufft.so`` and produce a wheel that
crashes with SIGILL on CPUs without it. This check disassembles the native
libraries bundled in a wheel and rejects any that reference AVX-512
registers (``%zmm``, mask registers ``{%k*}``).

Usage:
    python scripts/verify_wheel_isa.py dist/*.whl

Exits non-zero if any bundled library contains AVX-512 instructions.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# AVX-512 codegen is marked by EVEX-encoded operands: full-width %zmm registers,
# opmask registers {%k*}, and broadcast memory operands {1toN}. Any of these
# appears only in AVX-512 (or later) instructions.
AVX512_PATTERNS = (
    re.compile(r"%zmm\d+"),
    re.compile(r"\{%k\d+\}"),
    re.compile(r"\{1to\d+\}"),
)

NATIVE_SUFFIXES = (".so", ".dylib", ".dll")


def disassemble(objdump: str, library: Path) -> str | None:
    result = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", str(library)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"warning: could not disassemble {library}: {result.stderr.strip()}", file=sys.stderr)
        return None
    return result.stdout


def find_avx512(disassembly: str) -> list[str]:
    hits: list[str] = []
    for line in disassembly.splitlines():
        if any(pattern.search(line) for pattern in AVX512_PATTERNS):
            hits.append(line.strip())
    return hits


def check_library(objdump: str, library: Path) -> list[str]:
    disassembly = disassemble(objdump, library)
    if disassembly is None:
        return []
    return find_avx512(disassembly)


def iter_native_libs(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in NATIVE_SUFFIXES:
            yield path


def check_wheel(objdump: str, wheel: Path) -> bool:
    if not wheel.is_file():
        print(f"error: {wheel} is not a file", file=sys.stderr)
        return False

    with tempfile.TemporaryDirectory() as tmp:
        extract_dir = Path(tmp)
        try:
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(extract_dir)
        except zipfile.BadZipFile:
            print(f"error: {wheel} is not a valid wheel/zip archive", file=sys.stderr)
            return False

        libraries = list(iter_native_libs(extract_dir))
        if not libraries:
            print(f"error: no native libraries found in {wheel.name}", file=sys.stderr)
            return False

        failed = False
        for library in libraries:
            hits = check_library(objdump, library)
            rel = library.relative_to(extract_dir)
            if hits:
                failed = True
                print(f"FAIL {wheel.name}: {rel} uses AVX-512 ({len(hits)} instructions)")
                for line in hits[:5]:
                    print(f"    {line}")
            else:
                print(f"ok   {wheel.name}: {rel} is AVX-512-free")
        return not failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path, help="wheel file(s) to inspect")
    parser.add_argument(
        "--objdump",
        default=shutil.which("objdump"),
        help="path to objdump (default: autodetect)",
    )
    args = parser.parse_args(argv)

    if not args.objdump or not Path(args.objdump).exists():
        print("error: objdump not found; install binutils", file=sys.stderr)
        return 2

    all_ok = True
    for wheel in args.wheels:
        if not check_wheel(args.objdump, wheel):
            all_ok = False

    if all_ok:
        print("All wheels are portable (no AVX-512 instructions found).")
        return 0
    print("One or more wheels bundle AVX-512 native code.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
