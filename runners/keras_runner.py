from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any
import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from data import (
    ObservableDataset,
    ParticleDataset,
    load_split_entries,
    compute_observable_standardization,
)
from models.efn import EFN, MomentEFN, ObservableEFN, AttentionEFN


class ObservableMLP(nn.Module):
    """
    Simple observable baseline for x/nsubs/eecs/efps.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class OEFNDataset(Dataset):
    """
    Dataset wrapper for observable-enhanced EFN.

    Combines:
        ParticleDataset    -> particles, mask
        ObservableDataset  -> observables
    """

    def __init__(
        self,
        split_path: str | Path,
        split: str,
        fold: int | None,
        observable_key: str,
        max_particles: int,
        sort_by_pt: bool,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.parts_ds = ParticleDataset.from_split_json(
            split_path=split_path,
            split=split,
            fold=fold,
            max_particles=max_particles,
            sort_by_pt=sort_by_pt,
            return_metadata=False,
        )

        self.obsv_ds = ObservableDataset.from_split_json(
            split_path=split_path,
            split=split,
            fold=fold,
            observable_key=observable_key,
            mean=mean,
            std=std,
            return_metadata=False,
        )

        if len(self.parts_ds) != len(self.obsv_ds):
            raise ValueError(
                "ParticleDataset and ObservableDataset have different lengths: "
                f"{len(self.parts_ds)} vs {len(self.obsv_ds)}"
            )

    def __len__(self) -> int:
        return len(self.parts_ds)

    def __getitem__(self, idx: int):
        particles, mask, y, w = self.parts_ds[idx]
        x, y2, w2 = self.obsv_ds[idx]

        if int(y.item()) != int(y2.item()):
            raise ValueError("Label mismatch between particle and observable dataset.")

        return particles, mask, x, y, w


def save_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_device(device_config: str = "auto") -> torch.device:
    if device_config == "cpu":
        return torch.device("cpu")

    if device_config == "cuda":
        return torch.device("cuda")

    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raise ValueError(f"Unknown device setting: {device_config}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: dict[str, Any], input_dim: int | None = None) -> nn.Module:
    model_name = config["model"]["name"].lower()
    model_cfg = config["model"]

    if model_name == "mlp":
        if input_dim is None:
            raise ValueError("input_dim is required for mlp.")

        return ObservableMLP(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.20)),
        )

    if model_name == "efn":
        return EFN(
            phi_hidden_dims=model_cfg.get("phi_hidden_dims", [100, 100]),
            latent_dim=int(model_cfg.get("latent_dim", 126)),
            f_hidden_dims=model_cfg.get("f_hidden_dims", [100, 100, 100]),
            dropout=float(model_cfg.get("dropout", 0.075)),
        )

    if model_name == "mefn":
        return MomentEFN(
            latent_dim=int(model_cfg.get("latent_dim", 16)),
            moment_order=int(model_cfg.get("moment_order", 3)),
            phi_hidden_dims=model_cfg.get("phi_hidden_dims", [100, 100]),
            f_hidden_dims=model_cfg.get("f_hidden_dims", [100, 100, 100]),
            dropout=float(model_cfg.get("dropout", 0.20)),
        )

    if model_name == "oefn":
        if input_dim is None:
            raise ValueError("input_dim is required for oefn.")

        return ObservableEFN(
            observable_dim=input_dim,
            phi_hidden_dims=model_cfg.get("phi_hidden_dims", [100, 100]),
            latent_dim=int(model_cfg.get("latent_dim", 126)),
            f_hidden_dims=model_cfg.get("f_hidden_dims", [100, 100, 100]),
            dropout=float(model_cfg.get("dropout", 0.20)),
        )

    if model_name == "aefn":
        return AttentionEFN(
            phi_hidden_dims=model_cfg.get("phi_hidden_dims", [100, 100]),
            attention_dim=int(model_cfg.get("attention_dim", 128)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            num_attention_blocks=int(model_cfg.get("num_attention_blocks", 1)),
            f_hidden_dims=model_cfg.get("f_hidden_dims", [100, 100, 100]),
            dropout=float(model_cfg.get("dropout", 0.10)),
        )

    raise ValueError(f"Unknown model name for torch runner: {model_name}")


def make_datasets(config: dict[str, Any], fold: int):
    split_path = config["split"]["split_file"]
    input_type = config["input"]["type"].lower()
    observable_key = config["input"]["observable_key"]
    max_particles = int(config["input"]["max_particles"])
    sort_by_pt = bool(config["input"].get("sort_by_pt", True))

    train_entries = load_split_entries(split_path, split="train", fold=fold)

    if input_type == "obsv":
        mean, std = compute_observable_standardization(
            train_entries,
            observable_key=observable_key,
        )

        train_ds = ObservableDataset.from_split_json(
            split_path=split_path,
            split="train",
            fold=fold,
            observable_key=observable_key,
            mean=mean,
            std=std,
        )

        val_ds = ObservableDataset.from_split_json(
            split_path=split_path,
            split="val",
            fold=fold,
            observable_key=observable_key,
            mean=mean,
            std=std,
        )

        test_ds = ObservableDataset.from_split_json(
            split_path=split_path,
            split="test",
            fold=None,
            observable_key=observable_key,
            mean=mean,
            std=std,
        )

        return train_ds, val_ds, test_ds, int(mean.shape[0])

    if input_type == "parts":
        train_ds = ParticleDataset.from_split_json(
            split_path=split_path,
            split="train",
            fold=fold,
            max_particles=max_particles,
            sort_by_pt=sort_by_pt,
        )

        val_ds = ParticleDataset.from_split_json(
            split_path=split_path,
            split="val",
            fold=fold,
            max_particles=max_particles,
            sort_by_pt=sort_by_pt,
        )

        test_ds = ParticleDataset.from_split_json(
            split_path=split_path,
            split="test",
            fold=None,
            max_particles=max_particles,
            sort_by_pt=sort_by_pt,
        )

        return train_ds, val_ds, test_ds, None

    if input_type == "oefn":
        mean, std = compute_observable_standardization(
            train_entries,
            observable_key=observable_key,
        )

        train_ds = OEFNDataset(
            split_path=split_path,
            split="train",
            fold=fold,
            observable_key=observable_key,
            max_particles=max_particles,
            sort_by_pt=sort_by_pt,
            mean=mean,
            std=std,
        )

        val_ds = OEFNDataset(
            split_path=split_path,
            split="val",
            fold=fold,
            observable_key=observable_key,
            max_particles=max_particles,
            sort_by_pt=sort_by_pt,
            mean=mean,
            std=std,
        )

        test_ds = OEFNDataset(
            split_path=split_path,
            split="test",
            fold=None,
            observable_key=observable_key,
            max_particles=max_particles,
            sort_by_pt=sort_by_pt,
            mean=mean,
            std=std,
        )

        return train_ds, val_ds, test_ds, int(mean.shape[0])

    raise ValueError(f"Unknown input type: {input_type}")


def batch_to_device(batch, input_type: str, device: torch.device):
    if input_type == "obsv":
        x, y, w = batch
        return x.to(device), y.to(device), w.to(device)

    if input_type == "parts":
        particles, mask, y, w = batch
        return particles.to(device), mask.to(device), y.to(device), w.to(device)

    if input_type == "oefn":
        particles, mask, x, y, w = batch
        return (
            particles.to(device),
            mask.to(device),
            x.to(device),
            y.to(device),
            w.to(device),
        )

    raise ValueError(f"Unknown input type: {input_type}")


def forward_model(
    model: nn.Module,
    batch,
    input_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_type == "obsv":
        x, y, w = batch
        logits = model(x)
        return logits, y, w

    if input_type == "parts":
        particles, mask, y, w = batch
        logits = model(particles, mask)
        return logits, y, w

    if input_type == "oefn":
        particles, mask, x, y, w = batch
        logits = model(particles, mask, x)
        return logits, y, w

    raise ValueError(f"Unknown input type: {input_type}")


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    y_pred = (probs >= 0.5).astype(int)

    metrics: dict[str, float] = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics[f"{prefix}_roc_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics[f"{prefix}_tn"] = int(tn)
    metrics[f"{prefix}_fp"] = int(fp)
    metrics[f"{prefix}_fn"] = int(fn)
    metrics[f"{prefix}_tp"] = int(tp)

    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    input_type: str,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()

    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, input_type, device)
            logits, y, w = forward_model(model, batch, input_type)

            loss_raw = loss_fn(logits, y)
            loss = (loss_raw * w).mean()

            probs = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_labels).astype(int)
    probs = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, probs, prefix="eval")
    metrics["eval_loss"] = float(np.mean(losses))

    return metrics, y_true, probs


def run_one_fold(
    config: dict[str, Any],
    fold: int,
    run_dir: Path,
) -> dict[str, Any]:
    input_type = config["input"]["type"].lower()
    training_cfg = config["training"]

    device = get_device(training_cfg.get("device", "auto"))

    train_ds, val_ds, test_ds, input_dim = make_datasets(config, fold=fold)

    batch_size = int(training_cfg["batch_size"])
    num_workers = int(training_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config, input_dim=input_dim).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    epochs = int(training_cfg["epochs"])
    patience = int(training_cfg.get("patience", 10))

    fold_dir = run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = -np.inf
    best_epoch = -1
    patience_counter = 0
    history_rows: list[dict[str, Any]] = []

    print("=" * 100)
    print(f"Fold {fold}")
    print("Device:", device)
    print("Model:", config["model"]["name"])
    print("Input type:", input_type)
    print("Train jets:", len(train_ds))
    print("Val jets:", len(val_ds))
    print("Test jets:", len(test_ds))
    print("Parameters:", count_parameters(model))
    print("=" * 100)

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            batch = batch_to_device(batch, input_type, device)

            optimizer.zero_grad()

            logits, y, w = forward_model(model, batch, input_type)
            loss_raw = loss_fn(logits, y)
            loss = (loss_raw * w).mean()

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            input_type=input_type,
            device=device,
            loss_fn=loss_fn,
        )

        train_loss = float(np.mean(train_losses))
        val_auc = float(val_metrics.get("eval_roc_auc", float("nan")))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_metrics['eval_loss']:.5f} | "
            f"val_auc={val_auc:.5f}"
        )

        improved = not np.isnan(val_auc) and val_auc > best_val_auc

        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), fold_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    train_runtime = time.perf_counter() - start_time

    save_rows_csv(history_rows, fold_dir / "trainer_log_history.csv")

    best_model_path = fold_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics, y_true, probs = evaluate(
        model=model,
        loader=test_loader,
        input_type=input_type,
        device=device,
        loss_fn=loss_fn,
    )

    prediction_rows = []
    for i, (label, prob) in enumerate(zip(y_true, probs)):
        prediction_rows.append(
            {
                "index": i,
                "true_label": int(label),
                "prob_PbPb": float(prob),
                "prob_pp": float(1.0 - prob),
                "pred_label": int(prob >= 0.5),
            }
        )

    save_rows_csv(prediction_rows, fold_dir / "test_predictions.csv")

    result: dict[str, Any] = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "train_runtime": float(train_runtime),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "num_parameters": count_parameters(model),
    }

    result.update(
        {key.replace("eval_", "test_"): value for key, value in test_metrics.items()}
    )

    save_rows_csv([result], fold_dir / "test_metrics.csv")

    print("Fold result:")
    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def summarize_folds(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = sorted(
        {
            key
            for row in fold_rows
            for key in row.keys()
            if key.startswith("test_") or key in {"best_val_auc", "train_runtime"}
        }
    )

    summary_rows: list[dict[str, Any]] = []

    for metric in metric_names:
        values = [
            row[metric]
            for row in fold_rows
            if metric in row and isinstance(row[metric], (int, float))
        ]

        if not values:
            continue

        arr = np.asarray(values, dtype=float)

        summary_rows.append(
            {
                "metric": metric,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            }
        )

    return summary_rows


def run_torch_training(config: dict[str, Any]) -> None:
    output_root = Path(config["outputs"]["root"])
    model_name = config["model"]["name"].lower()

    run_id = datetime.now().strftime(f"{model_name}_run_%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_folds = int(config["split"]["n_folds"])
    fold_rows: list[dict[str, Any]] = []

    for fold in range(n_folds):
        fold_result = run_one_fold(
            config=config,
            fold=fold,
            run_dir=run_dir,
        )
        fold_rows.append(fold_result)

    save_rows_csv(fold_rows, run_dir / "fold_results.csv")

    summary_rows = summarize_folds(fold_rows)
    save_rows_csv(summary_rows, run_dir / "cv_summary.csv")

    print("=" * 100)
    print("CV SUMMARY")
    print("=" * 100)

    for row in summary_rows:
        print(f"{row['metric']}: {row['mean']:.5f} ± {row['std']:.5f}")

    print("Saved run to:", run_dir)
