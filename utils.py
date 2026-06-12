from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import random

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load YAML config file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """
    Save dictionary as formatted JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load JSON file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if needed and return Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_output_dirs(config: dict[str, Any]) -> None:
    """
    Create output directories from config.
    """
    outputs = config.get("outputs", {})

    for key in [
        "root",
        "checkpoints_dir",
        "logs_dir",
        "plots_dir",
        "predictions_dir",
    ]:
        if key in outputs:
            ensure_dir(outputs[key])

    split_cfg = config.get("split", {})
    if "output_dir" in split_cfg:
        ensure_dir(split_cfg["output_dir"])


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and optionally PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except ImportError:
        pass


def get_device(device_config: str = "auto") -> str:
    """
    Resolve training device.

    device_config:
        auto, cpu, cuda
    """
    if device_config == "cpu":
        return "cpu"

    if device_config == "cuda":
        return "cuda"

    if device_config != "auto":
        raise ValueError(f"Unknown device config: {device_config}")

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    except ImportError:
        return "cpu"


def get_data_root(config: dict[str, Any], override: str | None = None) -> Path:
    """
    Get dataset root from CLI override or config.
    """
    if override is not None:
        return Path(override)

    return Path(config["data"]["root"])


def get_label_from_group(group: str) -> int:
    """
    Map technical dataset group to implementation label.
    """
    group = group.lower()

    if group == "vac":
        return 0

    if group == "rec":
        return 1

    raise ValueError(f"Unknown group: {group}")


def get_physics_name_from_group(group: str) -> str:
    """
    Map technical dataset group to physics label.
    """
    group = group.lower()

    if group == "vac":
        return "pp"

    if group == "rec":
        return "PbPb"

    raise ValueError(f"Unknown group: {group}")


def print_config_summary(config: dict[str, Any]) -> None:
    """
    Print compact config summary.
    """
    print("=" * 100)
    print("CONFIG SUMMARY")
    print("=" * 100)

    print("Project:", config.get("project", {}).get("name", "unknown"))
    print("Data root:", config.get("data", {}).get("root", "missing"))

    input_cfg = config.get("input", {})
    print("Input type:", input_cfg.get("type", "missing"))
    print("Observable key:", input_cfg.get("observable_key", "missing"))
    print("Max particles:", input_cfg.get("max_particles", "missing"))

    split_cfg = config.get("split", {})
    print("Split seed:", split_cfg.get("seed", "missing"))
    print("Test size:", split_cfg.get("test_size", "missing"))
    print("Number of folds:", split_cfg.get("n_folds", "missing"))

    model_cfg = config.get("model", {})
    print("Model:", model_cfg.get("name", "missing"))
    print("Backend:", model_cfg.get("backend", "missing"))

    training_cfg = config.get("training", {})
    print("Batch size:", training_cfg.get("batch_size", "missing"))
    print("Epochs:", training_cfg.get("epochs", "missing"))
    print("Learning rate:", training_cfg.get("learning_rate", "missing"))
    print("=" * 100)


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    """
    Count PyTorch model parameters.
    """
    try:
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        return sum(p.numel() for p in model.parameters())

    except AttributeError as exc:
        raise TypeError("count_parameters expects a PyTorch-like model.") from exc
