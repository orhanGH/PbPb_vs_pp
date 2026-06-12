from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .splits import load_split_entries


def _get_first_existing_key(data: Any, candidates: list[str], path: Path) -> str:
    for key in candidates:
        if key in data:
            return key

    raise KeyError(
        f"None of the candidate keys {candidates} found in {path}. "
        f"Available keys: {list(data.keys())}"
    )


def _wrap_delta_phi(delta_phi: np.ndarray) -> np.ndarray:
    return (delta_phi + np.pi) % (2.0 * np.pi) - np.pi


class ParticleDataset(Dataset):
    """
    PyTorch Dataset for particle-level files.

    It builds per-jet particle tokens:

        (z_i, delta_eta_i, delta_phi_i)

    where:

        z_i = pT_i / sum_j pT_j

    The dataset supports variable-length constituent arrays stored with offsets.

    Expected approximate keys:

        pT / pt
        eta
        phi
        m
        offsets
        w

    Because the exact key naming can differ, this loader tries several common key names.
    """

    def __init__(
        self,
        entries: list[dict[str, Any]],
        max_particles: int = 128,
        sort_by_pt: bool = True,
        return_metadata: bool = False,
    ) -> None:
        self.entries = entries
        self.max_particles = max_particles
        self.sort_by_pt = sort_by_pt
        self.return_metadata = return_metadata

        self.index: list[tuple[int, int]] = []

        for file_id, entry in enumerate(self.entries):
            path = Path(entry["parts_path"])

            with np.load(path, allow_pickle=False) as data:
                offsets_key = _get_first_existing_key(
                    data,
                    ["offsets", "offset", "jet_offsets"],
                    path,
                )
                offsets = data[offsets_key]

                n_jets = len(offsets) - 1

            for local_idx in range(n_jets):
                self.index.append((file_id, local_idx))

        self._cache_file_id: int | None = None
        self._cache_data: Any | None = None
        self._cache_keys: dict[str, str] | None = None

    @classmethod
    def from_split_json(
        cls,
        split_path: str | Path,
        split: str,
        fold: int | None = None,
        max_particles: int = 128,
        sort_by_pt: bool = True,
        return_metadata: bool = False,
    ) -> "ParticleDataset":
        entries = load_split_entries(split_path=split_path, split=split, fold=fold)

        return cls(
            entries=entries,
            max_particles=max_particles,
            sort_by_pt=sort_by_pt,
            return_metadata=return_metadata,
        )

    def __len__(self) -> int:
        return len(self.index)

    def _load_file(self, file_id: int):
        if self._cache_file_id == file_id and self._cache_data is not None:
            return self._cache_data, self._cache_keys

        if self._cache_data is not None:
            self._cache_data.close()

        entry = self.entries[file_id]
        path = Path(entry["parts_path"])
        data = np.load(path, allow_pickle=False)

        keys = {
            "pt": _get_first_existing_key(data, ["pT", "pt", "pts", "particle_pt"], path),
            "eta": _get_first_existing_key(data, ["eta", "etas", "particle_eta"], path),
            "phi": _get_first_existing_key(data, ["phi", "phis", "particle_phi"], path),
            "offsets": _get_first_existing_key(
                data,
                ["offsets", "offset", "jet_offsets"],
                path,
            ),
        }

        if "w" in data:
            keys["w"] = "w"
        elif "weights" in data:
            keys["w"] = "weights"

        # Optional jet-axis keys. If absent, we compute a pT-weighted axis from constituents.
        if "jeta" in data:
            keys["jet_eta"] = "jeta"
        elif "jet_eta" in data:
            keys["jet_eta"] = "jet_eta"
        elif "eta_jet" in data:
            keys["jet_eta"] = "eta_jet"

        if "jphi" in data:
            keys["jet_phi"] = "jphi"
        elif "jet_phi" in data:
            keys["jet_phi"] = "jet_phi"
        elif "phi_jet" in data:
            keys["jet_phi"] = "phi_jet"

        if "jpt" in data:
            keys["jet_pt"] = "jpt"
        elif "jet_pt" in data:
            keys["jet_pt"] = "jet_pt"

        self._cache_data = data
        self._cache_keys = keys
        self._cache_file_id = file_id

        return data, keys

    def _get_constituents(
        self,
        data: Any,
        keys: dict[str, str],
        local_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offsets = data[keys["offsets"]]
        start = int(offsets[local_idx])
        end = int(offsets[local_idx + 1])

        pt = data[keys["pt"]][start:end].astype(np.float32)
        eta = data[keys["eta"]][start:end].astype(np.float32)
        phi = data[keys["phi"]][start:end].astype(np.float32)

        return pt, eta, phi

    def _compute_jet_axis(
        self,
        pt: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
    ) -> tuple[float, float, float]:
        pt_sum = float(np.sum(pt))

        if pt_sum <= 0.0:
            return 0.0, 0.0, 0.0

        z = pt / pt_sum

        jet_eta = float(np.sum(z * eta))

        # Circular pT-weighted mean for phi
        sin_phi = float(np.sum(z * np.sin(phi)))
        cos_phi = float(np.sum(z * np.cos(phi)))
        jet_phi = float(np.arctan2(sin_phi, cos_phi))

        return pt_sum, jet_eta, jet_phi

    def __getitem__(self, idx: int):
        file_id, local_idx = self.index[idx]
        entry = self.entries[file_id]

        data, keys = self._load_file(file_id)
        assert keys is not None

        pt, eta, phi = self._get_constituents(data, keys, local_idx)

        if pt.size == 0:
            tokens = np.zeros((self.max_particles, 3), dtype=np.float32)
            mask = np.zeros((self.max_particles,), dtype=np.float32)
            y = np.float32(entry["label"])
            w = np.float32(1.0)

            return (
                torch.from_numpy(tokens),
                torch.from_numpy(mask),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(w, dtype=torch.float32),
            )

        if "jet_pt" in keys:
            jet_pt = float(data[keys["jet_pt"]][local_idx])
        else:
            jet_pt = float(np.sum(pt))

        if "jet_eta" in keys and "jet_phi" in keys:
            jet_eta = float(data[keys["jet_eta"]][local_idx])
            jet_phi = float(data[keys["jet_phi"]][local_idx])
        else:
            _, jet_eta, jet_phi = self._compute_jet_axis(pt, eta, phi)

        if jet_pt <= 0.0:
            z = np.zeros_like(pt, dtype=np.float32)
        else:
            z = (pt / jet_pt).astype(np.float32)

        delta_eta = (eta - jet_eta).astype(np.float32)
        delta_phi = _wrap_delta_phi(phi - jet_phi).astype(np.float32)

        tokens = np.stack([z, delta_eta, delta_phi], axis=1).astype(np.float32)

        if self.sort_by_pt:
            order = np.argsort(-pt)
            tokens = tokens[order]

        n_particles = min(tokens.shape[0], self.max_particles)

        padded = np.zeros((self.max_particles, 3), dtype=np.float32)
        mask = np.zeros((self.max_particles,), dtype=np.float32)

        padded[:n_particles] = tokens[:n_particles]
        mask[:n_particles] = 1.0

        y = np.float32(entry["label"])

        if "w" in keys:
            w = data[keys["w"]][local_idx].astype(np.float32)
        else:
            w = np.float32(1.0)

        tokens_tensor = torch.from_numpy(padded)
        mask_tensor = torch.from_numpy(mask)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        w_tensor = torch.tensor(w, dtype=torch.float32)

        if not self.return_metadata:
            return tokens_tensor, mask_tensor, y_tensor, w_tensor

        metadata = {
            "file_name": entry["name"],
            "group": entry["group"],
            "physics_label": entry["physics_label"],
            "local_idx": local_idx,
            "n_particles": int(pt.size),
            "jet_pt": float(jet_pt),
            "jet_eta": float(jet_eta),
            "jet_phi": float(jet_phi),
        }

        return tokens_tensor, mask_tensor, y_tensor, w_tensor, metadata

    def __del__(self) -> None:
        if getattr(self, "_cache_data", None) is not None:
            self._cache_data.close()
