from __future__ import annotations

import argparse

from data.splits import make_file_splits


DEFAULT_DATA_ROOT = "/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create file-level train/val/test splits for PbPb_vs_pp."
    )

    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Path to the Marvin dataset root.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/splits",
        help="Directory where file_splits.json will be saved.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified splitting.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of files reserved for final test set.",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help="Number of cross-validation folds on the development set.",
    )

    args = parser.parse_args()

    make_file_splits(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        test_size=args.test_size,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
