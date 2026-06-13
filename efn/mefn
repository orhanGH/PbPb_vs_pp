from __future__ import annotations

import itertools
import torch
from torch import nn

from .efn import build_mlp


def make_moment_indices(latent_dim: int, moment_order: int) -> list[tuple[int, ...]]:
    indices: list[tuple[int, ...]] = []

    for order in range(1, moment_order + 1):
        indices.extend(
            itertools.combinations_with_replacement(range(latent_dim), order)
        )

    return indices


class MomentEFN(nn.Module):
    """
    Moment EFN adapted to the new PyTorch dataset.

    Input:
        particles: (batch, max_particles, 3)
        mask:      (batch, max_particles)

    Uses moment pooling over Phi(delta_eta, delta_phi):
        sum_i z_i * Phi_a(...)
        sum_i z_i * Phi_a(...) Phi_b(...)
        ...
    """

    def __init__(
        self,
        latent_dim: int = 16,
        moment_order: int = 3,
        phi_hidden_dims: list[int] | tuple[int, ...] = (100, 100),
        f_hidden_dims: list[int] | tuple[int, ...] = (100, 100, 100),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        if moment_order < 1:
            raise ValueError("moment_order must be >= 1")

        self.latent_dim = int(latent_dim)
        self.moment_order = int(moment_order)
        self.moment_indices = make_moment_indices(self.latent_dim, self.moment_order)
        self.effective_dim = len(self.moment_indices)

        self.phi = build_mlp(
            input_dim=2,
            hidden_dims=phi_hidden_dims,
            output_dim=self.latent_dim,
            dropout=dropout,
        )

        self.f = build_mlp(
            input_dim=self.effective_dim,
            hidden_dims=f_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def moment_pool(
        self,
        phi_out: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            phi_out = phi_out * mask_expanded
            z = z * mask_expanded

        pooled_moments = []

        for index_tuple in self.moment_indices:
            product = torch.ones_like(phi_out[..., 0])

            for latent_index in index_tuple:
                product = product * phi_out[..., latent_index]

            value = torch.sum(z.squeeze(-1) * product, dim=1)
            pooled_moments.append(value)

        return torch.stack(pooled_moments, dim=-1)

    def encode(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = particles[..., 0:1]
        coords = particles[..., 1:3]

        phi_out = self.phi(coords)
        moments = self.moment_pool(phi_out, z, mask)

        return moments

    def forward(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        moments = self.encode(particles, mask)
        logits = self.f(moments).squeeze(-1)
        return logits


def get_default_config() -> dict:
    return {
        "model_name": "mefn",
        "input_type": "parts",
        "latent_dim": 16,
        "moment_order": 3,
        "phi_hidden_dims": [100, 100],
        "f_hidden_dims": [100, 100, 100],
        "dropout": 0.20,
        "batch_size": 512,
        "epochs": 50,
        "patience": 10,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    }


def build_model(config: dict) -> MomentEFN:
    return MomentEFN(
        latent_dim=int(config.get("latent_dim", 16)),
        moment_order=int(config.get("moment_order", 3)),
        phi_hidden_dims=config.get("phi_hidden_dims", [100, 100]),
        f_hidden_dims=config.get("f_hidden_dims", [100, 100, 100]),
        dropout=float(config.get("dropout", 0.20)),
    )


def get_model_summary_fields(config: dict) -> dict:
    return {
        "latent_dim": config.get("latent_dim", 16),
        "moment_order": config.get("moment_order", 3),
        "phi_hidden_dims": str(config.get("phi_hidden_dims", [100, 100])),
        "f_hidden_dims": str(config.get("f_hidden_dims", [100, 100, 100])),
        "dropout": config.get("dropout", 0.20),
    }
