from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput


def get_torch_activation(name: str) -> nn.Module:
    name = name.lower()

    if name == "relu":
        return nn.ReLU()

    if name == "gelu":
        return nn.GELU()

    if name in {"silu", "swish"}:
        return nn.SiLU()

    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)

    raise ValueError(f"Unsupported activation: {name}")


class FallbackMambaBlock(nn.Module):
    """
    Fallback block if transformers.MambaModel is not available.

    This keeps imports working locally. On Marvin, if HuggingFace Mamba is
    installed, ParticleMambaClassifier will use the real MambaModel.
    """

    def __init__(self, hidden_dim: int, dropout: float, activation: str) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            get_torch_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ParticleMambaClassifier(nn.Module):
    """
    Mamba sequence model adapted from qg to PbPb_vs_pp.

    Input:
        particles: (batch, max_particles, 3)
        mask:      (batch, max_particles)

    Mamba is order-sensitive. The ordering is controlled in ParticleDataset.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_hidden_layers: int = 4,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dropout: float = 0.10,
        activation: str = "silu",
        max_particles: int = 128,
    ) -> None:
        super().__init__()

        self.uses_real_mamba = False

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            get_torch_activation(activation),
            nn.Dropout(dropout),
        )

        try:
            from transformers import MambaConfig, MambaModel

            mamba_config = MambaConfig(
                vocab_size=4,
                hidden_size=hidden_dim,
                state_size=state_size,
                num_hidden_layers=num_hidden_layers,
                conv_kernel=conv_kernel,
                expand=expand,
                hidden_act=activation if activation in {"silu", "gelu", "relu"} else "silu",
                pad_token_id=0,
                bos_token_id=0,
                eos_token_id=0,
                use_cache=False,
            )

            self.mamba = MambaModel(mamba_config)
            self.uses_real_mamba = True
            self.fallback_blocks = None

        except Exception:
            self.mamba = None
            self.fallback_blocks = nn.ModuleList(
                [
                    FallbackMambaBlock(
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(num_hidden_layers)
                ]
            )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        if mask is None:
            mask = (particles.abs().sum(dim=-1) > 0).float()

        valid_mask = mask.long()

        x = self.input_projection(particles)
        x = x * mask.unsqueeze(-1)

        if self.uses_real_mamba and self.mamba is not None:
            outputs = self.mamba(
                inputs_embeds=x,
                attention_mask=valid_mask,
                use_cache=False,
            )
            encoded = outputs.last_hidden_state

        else:
            encoded = x
            assert self.fallback_blocks is not None

            for block in self.fallback_blocks:
                encoded = block(encoded)
                encoded = encoded * mask.unsqueeze(-1)

        encoded = self.final_norm(encoded)
        encoded = encoded * mask.unsqueeze(-1)

        z = particles[..., 0].clamp_min(0.0)
        weights = z * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        jet_repr = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(jet_repr).squeeze(-1)

        if labels is None:
            return logits

        loss = self.loss_fn(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits.unsqueeze(-1),
        )


def get_default_config() -> dict:
    return {
        "model_name": "mamba",
        "input_type": "parts",
        "hidden_dim": 128,
        "num_hidden_layers": 4,
        "state_size": 16,
        "conv_kernel": 4,
        "expand": 2,
        "dropout": 0.10,
        "activation": "silu",
        "max_particles": 128,
        "batch_size": 256,
        "epochs": 10,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
    }


def build_model(config: dict) -> ParticleMambaClassifier:
    return ParticleMambaClassifier(
        input_dim=3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_hidden_layers=int(config.get("num_hidden_layers", 4)),
        state_size=int(config.get("state_size", 16)),
        conv_kernel=int(config.get("conv_kernel", 4)),
        expand=int(config.get("expand", 2)),
        dropout=float(config.get("dropout", 0.10)),
        activation=config.get("activation", "silu"),
        max_particles=int(config.get("max_particles", 128)),
    )


def get_model_summary_fields(config: dict) -> dict:
    return {
        "hidden_dim": config.get("hidden_dim", 128),
        "num_hidden_layers": config.get("num_hidden_layers", 4),
        "state_size": config.get("state_size", 16),
        "conv_kernel": config.get("conv_kernel", 4),
        "expand": config.get("expand", 2),
        "dropout": config.get("dropout", 0.10),
        "activation": config.get("activation", "silu"),
        "max_particles": config.get("max_particles", 128),
    }
