#!/usr/bin/env python3
"""Download specified SeisBench datasets into SEISBENCH_DATA.

Usage:
  python install_seisbench_datasets.py Iquique PNW OBST2024

Env:
  SEISBENCH_DATA: base folder (defaults to ~/seis_data)
"""
import argparse
import os
import sys
from typing import List

try:
    import seisbench.data as sbd
except Exception as e:
    print("[error] Could not import seisbench. Install with: pip install seisbench", file=sys.stderr)
    raise


def download_dataset(name: str) -> None:
    if not hasattr(sbd, name):
        raise ValueError(f"Unknown SeisBench dataset '{name}'.")
    cls = getattr(sbd, name)
    ds = cls()
    print(f"[info] Downloading {name} into base {ds._storage.basepath}")  # noqa: SLF001
    ds.download()
    print(f"[ok] {name} ready")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="Dataset names, e.g., Iquique PNW OBST2024")
    args = ap.parse_args(argv)

    base = os.environ.get("SEISBENCH_DATA", os.path.expanduser("~/seis_data"))
    os.makedirs(base, exist_ok=True)
    print(f"[info] SEISBENCH_DATA={base}")

    failed = []
    for n in args.names:
        try:
            download_dataset(n)
        except Exception as e:
            failed.append((n, str(e)))
            print(f"[fail] {n}: {e}", file=sys.stderr)

    if failed:
        print("\nSome downloads failed:")
        for n, msg in failed:
            print(f" - {n}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

