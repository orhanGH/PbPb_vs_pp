from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any

from sklearn.model_selection import StratifiedKFold, train_test_split


LABEL_MAP = {
    "vac": 0,  # pp / vacuum-like reference
    "rec": 1,  # PbPb / reconstructed medium-affected
}


def collect_file_pairs(data_root: str | Path) -> list[dict[str, Any]]:
    """
    Collect paired observable-level and particle-level files.

    Expected Marvin dataset layout:

        data_root/
        ├── vac/
        │   ├── parts/
        │   └── obsvs/
        └── rec/
            ├── parts/
            └── obsvs/

    Returns one entry per paired file.
    """
    root = Path(data_root)

    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    entries: list[dict[str, Any]] = []

    for group in ["vac", "rec"]:
        label = LABEL_MAP[group]

        obsv_dir = root / group / "obsvs"
        parts_dir = root / group / "parts"

        if not obsv_dir.exists():
            raise FileNotFoundError(f"Missing observable directory: {obsv_dir}")
        if not parts_dir.exists():
            raise FileNotFoundError(f"Missing particle directory: {parts_dir}")

        obsv_files = sorted(obsv_dir.glob("*.npz"))
        parts_files = sorted(parts_dir.glob("*.npz"))

        obsv_by_name = {path.name: path for path in obsv_files}
        parts_by_name = {path.name: path for path in parts_files}

        common_names = sorted(set(obsv_by_name) & set(parts_by_name))

        missing_obsv = sorted(set(parts_by_name) - set(obsv_by_name))
        missing_parts = sorted(set(obsv_by_name) - set(parts_by_name))

        if missing_obsv:
            print(
                f"WARNING: {group}: {len(missing_obsv)} parts files have no matching obsv file."
            )
            print("Example:", missing_obsv[:5])

        if missing_parts:
            print(
                f"WARNING: {group}: {len(missing_parts)} obsv files have no matching parts file."
            )
            print("Example:", missing_parts[:5])

        for name in common_names:
            entries.append(
                {
                    "name": name,
                    "group": group,
                    "label": label,
                    "physics_label": "pp" if group == "vac" else "PbPb",
                    "obsv_path": str(obsv_by_name[name]),
                    "parts_path": str(parts_by_name[name]),
                }
            )

    if not entries:
        raise RuntimeError(f"No paired .npz files found under {root}")

    return entries


def summarize_entries(entries: list[dict[str, Any]], title: str = "entries") -> None:
    n_total = len(entries)
    n_vac = sum(entry["group"] == "vac" for entry in entries)
    n_rec = sum(entry["group"] == "rec" for entry in entries)

    print(f"{title}:")
    print(f"  total: {n_total}")
    print(f"  vac / pp: {n_vac}")
    print(f"  rec / PbPb: {n_rec}")


def make_file_splits(
    data_root: str | Path,
    output_dir: str | Path = "outputs/splits",
    seed: int = 42,
    test_size: float = 0.20,
    n_folds: int = 4,
) -> Path:
    """
    Create file-level train/validation/test splits.

    Important:
    Splitting is done at file level, not jet level. This avoids leakage from
    file-specific background conditions or preprocessing artifacts.
    """
    entries = collect_file_pairs(data_root)

    labels = [entry["label"] for entry in entries]

    dev_entries, test_entries = train_test_split(
        entries,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    dev_labels = [entry["label"] for entry in dev_entries]

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    folds: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(dev_entries, dev_labels)):
        train_entries = [dev_entries[i] for i in train_idx]
        val_entries = [dev_entries[i] for i in val_idx]

        folds.append(
            {
                "fold": fold_id,
                "train": train_entries,
                "val": val_entries,
            }
        )

    split_data = {
        "data_root": str(data_root),
        "seed": seed,
        "test_size": test_size,
        "n_folds": n_folds,
        "split_level": "file",
        "label_map": LABEL_MAP,
        "n_total_files": len(entries),
        "n_dev_files": len(dev_entries),
        "n_test_files": len(test_entries),
        "test": test_entries,
        "folds": folds,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "file_splits.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    print("Saved split file:", output_path)
    summarize_entries(entries, "all files")
    summarize_entries(dev_entries, "development files")
    summarize_entries(test_entries, "test files")

    for fold in folds:
        summarize_entries(fold["train"], f"fold {fold['fold']} train")
        summarize_entries(fold["val"], f"fold {fold['fold']} val")

    return output_path


def load_split_entries(
    split_path: str | Path,
    split: str,
    fold: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load entries from file_splits.json.

    split:
        "train", "val", or "test"

    fold:
        required for train/val
    """
    split_path = Path(split_path)

    with split_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if split == "test":
        return data["test"]

    if split in {"train", "val"}:
        if fold is None:
            raise ValueError("fold must be provided for train/val split")

        for fold_data in data["folds"]:
            if fold_data["fold"] == fold:
                return fold_data[split]

        raise ValueError(f"Fold {fold} not found in {split_path}")

    raise ValueError(f"Unknown split: {split}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-folds", type=int, default=4)

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
