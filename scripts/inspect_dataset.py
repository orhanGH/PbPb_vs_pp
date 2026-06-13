from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np


DEFAULT_DATA_ROOT = "/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50"


def inspect_file(path: Path) -> None:
    print("-" * 100)
    print(f"FILE: {path}")

    with np.load(path, allow_pickle=False) as data:
        print("KEYS:", list(data.keys()))

        for key in data.keys():
            arr = data[key]
            print(f"  {key:12s} shape={arr.shape} dtype={arr.dtype}")


def count_files(root: Path) -> None:
    print("=" * 100)
    print("FILE COUNTS")
    print("=" * 100)

    total = list(root.rglob("*.npz"))
    print("total .npz:", len(total))

    for group in ["vac", "rec"]:
        for kind in ["parts", "obsvs"]:
            files = sorted((root / group / kind).glob("*.npz"))
            print(f"{group}/{kind}: {len(files)}")


def inspect_dataset(data_root: str | Path, n: int = 2) -> None:
    root = Path(data_root)

    print("=" * 100)
    print("DATASET INSPECTION")
    print("=" * 100)
    print("data root:", root)
    print("exists:", root.exists())

    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    count_files(root)

    for group in ["vac", "rec"]:
        for kind in ["obsvs", "parts"]:
            files = sorted((root / group / kind).glob("*.npz"))

            print("=" * 100)
            print(f"{group}/{kind}: showing first {min(n, len(files))} files")

            for path in files[:n]:
                inspect_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PbPb_vs_pp Marvin .npz dataset."
    )

    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Path to the Marvin dataset root.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of files to inspect per group/subfolder.",
    )

    args = parser.parse_args()

    inspect_dataset(data_root=args.data_root, n=args.n)


if __name__ == "__main__":
    main()
