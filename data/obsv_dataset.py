from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .splits import load_split_entries


class ObservableDataset(Dataset):
    """
    PyTorch Dataset for observable-level files.

    Expected keys in each .npz file:

        x      shape = (N, 3472)
        w      shape = (N,)
        jpt    shape = (N,)
        jet    shape = (N,)
        nsubs  shape = (N, 60)
        eecs   shape = (N, 23)
        efps   shape = (N, 3389)

    The default observable_key is "x", i.e. all observables concatenated.
    """

    def __init__(
        self,
        entries: list[dict[str, Any]],
        observable_key: str = "x",
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.entries = entries
        self.observable_key = observable_key
        self.mean = mean
        self.std = std
        self.return_metadata = return_metadata

        self.index: list[tuple[int, int]] = []

        for file_id, entry in enumerate(self.entries):
            path = Path(entry["obsv_path"])

            with np.load(path, allow_pickle=False) as data:
                if observable_key not in data:
                    raise KeyError(
                        f"Key '{observable_key}' not found in {path}. "
                        f"Available keys: {list(data.keys())}"
                    )

                n_jets = data[observable_key].shape[0]

            for local_idx in range(n_jets):
                self.index.append((file_id, local_idx))

        self._cache_file_id: int | None = None
        self._cache_data: Any | None = None

    @classmethod
    def from_split_json(
        cls,
        split_path: str | Path,
        split: str,
        fold: int | None = None,
        observable_key: str = "x",
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        return_metadata: bool = False,
    ) -> "ObservableDataset":
        entries = load_split_entries(split_path=split_path, split=split, fold=fold)

        return cls(
            entries=entries,
            observable_key=observable_key,
            mean=mean,
            std=std,
            return_metadata=return_metadata,
        )

    def __len__(self) -> int:
        return len(self.index)

    def _load_file(self, file_id: int):
        if self._cache_file_id == file_id and self._cache_data is not None:
            return self._cache_data

        if self._cache_data is not None:
            self._cache_data.close()

        entry = self.entries[file_id]
        path = Path(entry["obsv_path"])

        self._cache_data = np.load(path, allow_pickle=False)
        self._cache_file_id = file_id

        return self._cache_data

    def __getitem__(self, idx: int):
        file_id, local_idx = self.index[idx]
        entry = self.entries[file_id]
        data = self._load_file(file_id)

        x = data[self.observable_key][local_idx].astype(np.float32)
        y = np.float32(entry["label"])

        if "w" in data:
            w = data["w"][local_idx].astype(np.float32)
        else:
            w = np.float32(1.0)

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        w_tensor = torch.tensor(w, dtype=torch.float32)

        if not self.return_metadata:
            return x_tensor, y_tensor, w_tensor

        metadata = {
            "file_name": entry["name"],
            "group": entry["group"],
            "physics_label": entry["physics_label"],
            "local_idx": local_idx,
        }

        if "jpt" in data:
            metadata["jpt"] = float(data["jpt"][local_idx])

        if "jet" in data:
            metadata["jet"] = float(data["jet"][local_idx])

        return x_tensor, y_tensor, w_tensor, metadata

    def __del__(self) -> None:
        if getattr(self, "_cache_data", None) is not None:
            self._cache_data.close()


def compute_observable_standardization(
    entries: list[dict[str, Any]],
    observable_key: str = "x",
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std from training files only.

    Use this only on train entries, then pass the resulting mean/std to
    train, validation, and test datasets.
    """
    n_total = 0
    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None

    for entry in entries:
        path = Path(entry["obsv_path"])

        with np.load(path, allow_pickle=False) as data:
            if observable_key not in data:
                raise KeyError(
                    f"Key '{observable_key}' not found in {path}. "
                    f"Available keys: {list(data.keys())}"
                )

            x = data[observable_key].astype(np.float64)

        if x.ndim != 2:
            raise ValueError(f"Expected 2D observable array, got shape {x.shape} in {path}")

        if sum_x is None:
            n_features = x.shape[1]
            sum_x = np.zeros(n_features, dtype=np.float64)
            sum_x2 = np.zeros(n_features, dtype=np.float64)

        n_total += x.shape[0]
        sum_x += x.sum(axis=0)
        sum_x2 += np.square(x).sum(axis=0)

    if n_total == 0 or sum_x is None or sum_x2 is None:
        raise RuntimeError("No jets found while computing observable standardization.")

    mean = sum_x / n_total
    var = sum_x2 / n_total - np.square(mean)
    std = np.sqrt(np.maximum(var, eps))

    return mean.astype(np.float32), std.astype(np.float32)


def save_standardization(
    output_path: str | Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, mean=mean, std=std)


def load_standardization(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["mean"].astype(np.float32), data["std"].astype(np.float32)
