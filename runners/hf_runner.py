from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any
import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from data import ParticleDataset
from models.hf import (
    ParticleBertClassifier,
    ParticleRobertaClassifier,
    ParticleMambaClassifier,
)


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


def build_hf_model(config: dict[str, Any]) -> nn.Module:
    name = config["model"]["name"].lower()
    model_cfg = config["model"]
    input_cfg = config["input"]

    max_particles = int(input_cfg.get("max_particles", 128))

    if name == "bert":
        return ParticleBertClassifier(
            input_dim=3,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_hidden_layers=int(model_cfg.get("num_hidden_layers", 4)),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
            intermediate_size=int(model_cfg.get("intermediate_size", 256)),
            dropout=float(model_cfg.get("dropout", 0.10)),
            activation=model_cfg.get("activation", "gelu"),
            max_particles=max_particles,
        )

    if name == "roberta":
        return ParticleRobertaClassifier(
            input_dim=3,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_hidden_layers=int(model_cfg.get("num_hidden_layers", 4)),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
            intermediate_size=int(model_cfg.get("intermediate_size", 256)),
            dropout=float(model_cfg.get("dropout", 0.10)),
            activation=model_cfg.get("activation", "gelu"),
            max_particles=max_particles,
        )

    if name == "mamba":
        return ParticleMambaClassifier(
            input_dim=3,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_hidden_layers=int(model_cfg.get("num_hidden_layers", 4)),
            state_size=int(model_cfg.get("state_size", model_cfg.get("d_state", 16))),
            conv_kernel=int(model_cfg.get("conv_kernel", model_cfg.get("d_conv", 4))),
            expand=int(model_cfg.get("expand", 2)),
            dropout=float(model_cfg.get("dropout", 0.10)),
            activation=model_cfg.get("activation", "silu"),
            max_particles=max_particles,
        )

    raise ValueError(f"Unknown HF/sequence model: {name}")


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
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()

    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for particles, mask, y, w in loader:
            particles = particles.to(device)
            mask = mask.to(device)
            y = y.to(device)
            w = w.to(device)

            logits = model(particles, mask)

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


def make_particle_datasets(config: dict[str, Any], fold: int):
    split_path = config["split"]["split_file"]
    max_particles = int(config["input"]["max_particles"])
    sort_by_pt = bool(config["input"].get("sort_by_pt", True))

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

    return train_ds, val_ds, test_ds


def run_one_hf_fold(
    config: dict[str, Any],
    fold: int,
    run_dir: Path,
) -> dict[str, Any]:
    training_cfg = config["training"]

    device = get_device(training_cfg.get("device", "auto"))

    train_ds, val_ds, test_ds = make_particle_datasets(config, fold=fold)

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

    model = build_hf_model(config).to(device)

    optimizer = torch.optim.AdamW(
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
    print(f"Sequence model fold {fold}")
    print("Device:", device)
    print("Model:", config["model"]["name"])
    print("Train jets:", len(train_ds))
    print("Val jets:", len(val_ds))
    print("Test jets:", len(test_ds))
    print("Parameters:", count_parameters(model))
    print("Uses real Mamba:", bool(getattr(model, "uses_real_mamba", False)))
    print("=" * 100)

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []

        for particles, mask, y, w in train_loader:
            particles = particles.to(device)
            mask = mask.to(device)
            y = y.to(device)
            w = w.to(device)

            optimizer.zero_grad()

            logits = model(particles, mask)

            loss_raw = loss_fn(logits, y)
            loss = (loss_raw * w).mean()

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
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
        "uses_real_mamba": bool(getattr(model, "uses_real_mamba", False)),
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


def run_hf_training(config: dict[str, Any]) -> None:
    output_root = Path(config["outputs"]["root"])
    model_name = config["model"]["name"].lower()

    run_id = datetime.now().strftime(f"{model_name}_run_%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_folds = int(config["split"]["n_folds"])

    fold_rows: list[dict[str, Any]] = []

    for fold in range(n_folds):
        fold_result = run_one_hf_fold(
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
