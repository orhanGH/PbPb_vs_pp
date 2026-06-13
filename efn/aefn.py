from __future__ import annotations

import torch
from torch import nn

from .efn import build_mlp


class AttentionEFN(nn.Module):
    """
    Attention EFN adapted from the qg aEFN idea.

    Structure:
        Phi(delta_eta, delta_phi)
        particle self-attention
        z-weighted sum
        F classifier

    Input:
        particles: (batch, max_particles, 3)
        mask:      (batch, max_particles)
    """

    def __init__(
        self,
        phi_hidden_dims: list[int] | tuple[int, ...] = (100, 100),
        attention_dim: int = 128,
        num_heads: int = 4,
        num_attention_blocks: int = 1,
        f_hidden_dims: list[int] | tuple[int, ...] = (100, 100, 100),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")

        self.attention_dim = int(attention_dim)

        self.phi = build_mlp(
            input_dim=2,
            hidden_dims=phi_hidden_dims,
            output_dim=self.attention_dim,
            dropout=dropout,
        )

        self.attention_blocks = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.ff_blocks = nn.ModuleList()
        self.norms_2 = nn.ModuleList()

        for _ in range(num_attention_blocks):
            self.attention_blocks.append(
                nn.MultiheadAttention(
                    embed_dim=self.attention_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )

            self.norms_1.append(nn.LayerNorm(self.attention_dim))

            self.ff_blocks.append(
                nn.Sequential(
                    nn.Linear(self.attention_dim, self.attention_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

            self.norms_2.append(nn.LayerNorm(self.attention_dim))

        self.f = build_mlp(
            input_dim=self.attention_dim,
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

        x = self.phi(coords)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask <= 0
            x = x * mask.unsqueeze(-1)

        for attn, norm1, ff, norm2 in zip(
            self.attention_blocks,
            self.norms_1,
            self.ff_blocks,
            self.norms_2,
        ):
            attn_out, _ = attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

            x = norm1(x + attn_out)

            ff_out = ff(x)
            x = norm2(x + ff_out)

            if mask is not None:
                x = x * mask.unsqueeze(-1)

        if mask is not None:
            z = z * mask.unsqueeze(-1)

        latent = torch.sum(z * x, dim=1)
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
        "model_name": "aefn",
        "input_type": "parts",
        "phi_hidden_dims": [100, 100],
        "attention_dim": 128,
        "num_heads": 4,
        "num_attention_blocks": 1,
        "f_hidden_dims": [100, 100, 100],
        "dropout": 0.10,
        "batch_size": 512,
        "epochs": 50,
        "patience": 10,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    }


def build_model(config: dict) -> AttentionEFN:
    return AttentionEFN(
        phi_hidden_dims=config.get("phi_hidden_dims", [100, 100]),
        attention_dim=int(config.get("attention_dim", 128)),
        num_heads=int(config.get("num_heads", 4)),
        num_attention_blocks=int(config.get("num_attention_blocks", 1)),
        f_hidden_dims=config.get("f_hidden_dims", [100, 100, 100]),
        dropout=float(config.get("dropout", 0.10)),
    )


def get_model_summary_fields(config: dict) -> dict:
    return {
        "phi_hidden_dims": str(config.get("phi_hidden_dims", [100, 100])),
        "attention_dim": config.get("attention_dim", 128),
        "num_heads": config.get("num_heads", 4),
        "num_attention_blocks": config.get("num_attention_blocks", 1),
        "f_hidden_dims": str(config.get("f_hidden_dims", [100, 100, 100])),
        "dropout": config.get("dropout", 0.10),
    }
