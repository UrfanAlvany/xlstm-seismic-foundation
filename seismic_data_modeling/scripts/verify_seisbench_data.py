#!/usr/bin/env python3
import argparse
import os
import sys


EXPECTED = [
    "ETHZ",
    "GEOFON",
    "STEAD",
    "INSTANCE",
    "Iquique",
    "PNW",
    "OBST2024",
    "MLAAPDE",
]

FILES = ["metadata.csv", "waveforms.hdf5"]


def check_dataset(base: str, name: str) -> list[str]:
    missing = []
    dpath = os.path.join(base, name)
    if not os.path.isdir(dpath):
        return [f"{name}: missing folder {dpath}"]
    for f in FILES:
        if not os.path.isfile(os.path.join(dpath, f)):
            missing.append(f"{name}: missing {f}")
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("SEISBENCH_DATA", os.path.expanduser("~/seis_data")), help="Base folder containing SeisBench dataset subfolders")
    ap.add_argument("--only", nargs="*", help="Optional list of dataset names to check")
    args = ap.parse_args()

    names = args.only if args.only else EXPECTED
    base = os.path.abspath(args.base)
    print(f"Checking SeisBench data under: {base}")

    all_missing: list[str] = []
    present: list[str] = []
    for n in names:
        problems = check_dataset(base, n)
        if problems:
            all_missing.extend(problems)
        else:
            present.append(n)

    if present:
        print("Present:", ", ".join(present))
    if all_missing:
        print("\nMissing components detected:")
        for m in all_missing:
            print(" -", m)
        print("\nTo download missing datasets with SeisBench, run (Python):")
        print("  >>> import seisbench.data as sbd")
        print("  >>> for name in [list...]: sbd.__getattribute__(name)().download()")
        return 1
    else:
        print("All expected datasets are present.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

