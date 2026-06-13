from __future__ import annotations

import torch
from torch import nn
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ParticleBertClassifier(nn.Module):
    """
    BERT encoder adapted from qg to PbPb_vs_pp.

    This is not a pretrained NLP model.
    It uses a BERT encoder initialized from scratch on particle tokens.

    Input:
        particles: (batch, max_particles, 3)
        mask:      (batch, max_particles)

    Output for our runners:
        logits: (batch,)

    If labels are passed, returns SequenceClassifierOutput for compatibility.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int | None = None,
        dropout: float = 0.10,
        activation: str = "gelu",
        max_particles: int = 128,
    ) -> None:
        super().__init__()

        if intermediate_size is None:
            intermediate_size = 4 * hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        config = BertConfig(
            vocab_size=2,
            hidden_size=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_particles,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            hidden_act=activation,
            type_vocab_size=1,
            num_labels=2,
        )

        self.bert = BertModel(config, add_pooling_layer=False)
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

        attention_mask = mask.long()

        inputs_embeds = self.input_projection(particles)

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        encoded = outputs.last_hidden_state

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
        "model_name": "bert",
        "input_type": "parts",
        "hidden_dim": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 256,
        "dropout": 0.10,
        "activation": "gelu",
        "max_particles": 128,
        "batch_size": 256,
        "epochs": 10,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
    }


def build_model(config: dict) -> ParticleBertClassifier:
    return ParticleBertClassifier(
        input_dim=3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_hidden_layers=int(config.get("num_hidden_layers", 4)),
        num_attention_heads=int(config.get("num_attention_heads", 4)),
        intermediate_size=int(config.get("intermediate_size", 256)),
        dropout=float(config.get("dropout", 0.10)),
        activation=config.get("activation", "gelu"),
        max_particles=int(config.get("max_particles", 128)),
    )


def get_model_summary_fields(config: dict) -> dict:
    return {
        "hidden_dim": config.get("hidden_dim", 128),
        "num_hidden_layers": config.get("num_hidden_layers", 4),
        "num_attention_heads": config.get("num_attention_heads", 4),
        "intermediate_size": config.get("intermediate_size", 256),
        "dropout": config.get("dropout", 0.10),
        "activation": config.get("activation", "gelu"),
        "max_particles": config.get("max_particles", 128),
    }
