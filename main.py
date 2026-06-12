from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    get_data_root,
    load_config,
    print_config_summary,
    set_seed,
    setup_output_dirs,
)


def run_inspect(data_root: Path, n: int) -> None:
    """
    Inspect dataset files and print keys/shapes.
    """
    import numpy as np

    print("=" * 100)
    print("DATASET INSPECTION")
    print("=" * 100)
    print("Data root:", data_root)
    print("Exists:", data_root.exists())

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    total_npz = list(data_root.rglob("*.npz"))
    print("Total .npz files:", len(total_npz))
    print()

    for group in ["vac", "rec"]:
        for kind in ["obsvs", "parts"]:
            files = sorted((data_root / group / kind).glob("*.npz"))

            print("=" * 100)
            print(f"{group}/{kind}: {len(files)} files")

            for path in files[:n]:
                print("-" * 100)
                print("FILE:", path)

                with np.load(path, allow_pickle=False) as data:
                    print("KEYS:", list(data.keys()))

                    for key in data.keys():
                        arr = data[key]
                        print(f"  {key:12s} shape={arr.shape} dtype={arr.dtype}")


def run_make_splits(config: dict, data_root: Path) -> None:
    """
    Create file-level splits.
    """
    from data.splits import make_file_splits

    split_cfg = config["split"]

    make_file_splits(
        data_root=data_root,
        output_dir=split_cfg["output_dir"],
        seed=int(split_cfg["seed"]),
        test_size=float(split_cfg["test_size"]),
        n_folds=int(split_cfg["n_folds"]),
    )


def run_check_obsv(config: dict) -> None:
    """
    Quick check for ObservableDataset.
    """
    from data import (
        ObservableDataset,
        compute_observable_standardization,
        load_split_entries,
    )

    split_path = config["split"]["split_file"]
    observable_key = config["input"]["observable_key"]

    if not Path(split_path).exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            "Run first: python main.py --mode make_splits"
        )

    print("=" * 100)
    print("OBSERVABLE DATASET CHECK")
    print("=" * 100)

    train_entries = load_split_entries(split_path, split="train", fold=0)

    print("Train files:", len(train_entries))
    print("Computing mean/std from train files only...")
    mean, std = compute_observable_standardization(
        train_entries,
        observable_key=observable_key,
    )

    train_ds = ObservableDataset.from_split_json(
        split_path=split_path,
        split="train",
        fold=0,
        observable_key=observable_key,
        mean=mean,
        std=std,
        return_metadata=True,
    )

    print("Number of train jets:", len(train_ds))

    sample = train_ds[0]
    x, y, w, metadata = sample

    print("Sample x shape:", tuple(x.shape))
    print("Sample y:", y.item())
    print("Sample w:", w.item())
    print("Sample metadata:", metadata)
    print("x mean approx after standardization:", float(x.mean()))
    print("x std approx after standardization:", float(x.std()))


def run_check_parts(config: dict) -> None:
    """
    Quick check for ParticleDataset.
    """
    from data import ParticleDataset

    split_path = config["split"]["split_file"]
    max_particles = int(config["input"]["max_particles"])
    sort_by_pt = bool(config["input"].get("sort_by_pt", True))

    if not Path(split_path).exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            "Run first: python main.py --mode make_splits"
        )

    print("=" * 100)
    print("PARTICLE DATASET CHECK")
    print("=" * 100)

    train_ds = ParticleDataset.from_split_json(
        split_path=split_path,
        split="train",
        fold=0,
        max_particles=max_particles,
        sort_by_pt=sort_by_pt,
        return_metadata=True,
    )

    print("Number of train jets:", len(train_ds))

    sample = train_ds[0]
    particles, mask, y, w, metadata = sample

    print("Particles shape:", tuple(particles.shape))
    print("Mask shape:", tuple(mask.shape))
    print("Number of valid particles:", int(mask.sum().item()))
    print("Sample y:", y.item())
    print("Sample w:", w.item())
    print("Sample metadata:", metadata)
    print("First 5 particles:")
    print(particles[:5])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PbPb_vs_pp project entry point"
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "inspect",
            "make_splits",
            "check_obsv",
            "check_parts",
            "summary",
        ],
        help="Action to run.",
    )

    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional override for dataset root.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of files per group/kind for inspect mode.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config.get("split", {}).get("seed", 42))
    set_seed(seed)

    setup_output_dirs(config)
    print_config_summary(config)

    data_root = get_data_root(config, override=args.data_root)

    if args.mode == "summary":
        return

    if args.mode == "inspect":
        run_inspect(data_root=data_root, n=args.n)
        return

    if args.mode == "make_splits":
        run_make_splits(config=config, data_root=data_root)
        return

    if args.mode == "check_obsv":
        run_check_obsv(config=config)
        return

    if args.mode == "check_parts":
        run_check_parts(config=config)
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
