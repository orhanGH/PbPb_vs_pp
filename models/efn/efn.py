from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dims: list[int] | tuple[int, ...],
    output_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    dropout: float = 0.0,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(activation())

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        prev_dim = int(hidden_dim)

    layers.append(nn.Linear(prev_dim, output_dim))

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)


class EFN(nn.Module):
    """
    Energy Flow Network adapted from the qg EFN idea.

    Input:
        particles: (batch, max_particles, 3)
            particles[..., 0] = z_i
            particles[..., 1] = delta_eta_i
            particles[..., 2] = delta_phi_i

        mask: (batch, max_particles)
            1 for valid particles, 0 for padding

    Output:
        logits: (batch,)
            binary logit for PbPb/rec class.
    """

    def __init__(
        self,
        phi_hidden_dims: list[int] | tuple[int, ...] = (100, 100),
        latent_dim: int = 126,
        f_hidden_dims: list[int] | tuple[int, ...] = (100, 100, 100),
        dropout: float = 0.075,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)

        self.phi = build_mlp(
            input_dim=2,
            hidden_dims=phi_hidden_dims,
            output_dim=self.latent_dim,
            dropout=dropout,
        )

        self.f = build_mlp(
            input_dim=self.latent_dim,
            hidden_dims=f_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def encode(
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
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.encode(particles, mask)
        logits = self.f(latent).squeeze(-1)
        return logits


def get_default_config() -> dict:
    return {
        "model_name": "efn",
        "input_type": "parts",
        "latent_dim": 126,
        "phi_hidden_dims": [100, 100],
        "f_hidden_dims": [100, 100, 100],
        "dropout": 0.075,
        "batch_size": 512,
        "epochs": 50,
        "patience": 10,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    }


def build_model(config: dict) -> EFN:
    return EFN(
        phi_hidden_dims=config.get("phi_hidden_dims", [100, 100]),
        latent_dim=int(config.get("latent_dim", 126)),
        f_hidden_dims=config.get("f_hidden_dims", [100, 100, 100]),
        dropout=float(config.get("dropout", 0.075)),
    )


def get_model_summary_fields(config: dict) -> dict:
    return {
        "latent_dim": config.get("latent_dim", 126),
        "phi_hidden_dims": str(config.get("phi_hidden_dims", [100, 100])),
        "f_hidden_dims": str(config.get("f_hidden_dims", [100, 100, 100])),
        "dropout": config.get("dropout", 0.075),
    }
