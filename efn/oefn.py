from __future__ import annotations

import torch
from torch import nn

from .efn import build_mlp


class ObservableEFN(nn.Module):
    """
    Observable-enhanced EFN.

    Combines:
        EFN latent representation from particle constituents
        + standardized high-level observables x/nsubs/eecs/efps.

    Input:
        particles:   (batch, max_particles, 3)
        mask:        (batch, max_particles)
        observables: (batch, observable_dim)

    Output:
        logits: (batch,)
    """

    def __init__(
        self,
        observable_dim: int = 3472,
        phi_hidden_dims: list[int] | tuple[int, ...] = (100, 100),
        latent_dim: int = 126,
        f_hidden_dims: list[int] | tuple[int, ...] = (100, 100, 100),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.observable_dim = int(observable_dim)
        self.latent_dim = int(latent_dim)

        self.phi = build_mlp(
            input_dim=2,
            hidden_dims=phi_hidden_dims,
            output_dim=self.latent_dim,
            dropout=dropout,
        )

        self.f = build_mlp(
            input_dim=self.latent_dim + self.observable_dim,
            hidden_dims=f_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def encode_particles(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = particles[..., 0:1]
        coords = particles[..., 1:3]

        phi_out = self.phi(coords)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            z = z * mask
            phi_out = phi_out * mask

        latent = torch.sum(z * phi_out, dim=1)
        return latent

    def forward(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor,
        observables: torch.Tensor,
    ) -> torch.Tensor:
        particle_latent = self.encode_particles(particles, mask)
        combined = torch.cat([particle_latent, observables], dim=-1)
        logits = self.f(combined).squeeze(-1)
        return logits


def get_default_config() -> dict:
    return {
        "model_name": "oefn",
        "input_type": "oefn",
        "observable_dim": 3472,
        "latent_dim": 126,
        "phi_hidden_dims": [100, 100],
        "f_hidden_dims": [100, 100, 100],
        "dropout": 0.20,
        "batch_size": 512,
        "epochs": 50,
        "patience": 10,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    }


def build_model(config: dict) -> ObservableEFN:
    return ObservableEFN(
        observable_dim=int(config.get("observable_dim", 3472)),
        phi_hidden_dims=config.get("phi_hidden_dims", [100, 100]),
        latent_dim=int(config.get("latent_dim", 126)),
        f_hidden_dims=config.get("f_hidden_dims", [100, 100, 100]),
        dropout=float(config.get("dropout", 0.20)),
    )


def get_model_summary_fields(config: dict) -> dict:
    return {
        "observable_dim": config.get("observable_dim", 3472),
        "latent_dim": config.get("latent_dim", 126),
        "phi_hidden_dims": str(config.get("phi_hidden_dims", [100, 100])),
        "f_hidden_dims": str(config.get("f_hidden_dims", [100, 100, 100])),
        "dropout": config.get("dropout", 0.20),
    }
