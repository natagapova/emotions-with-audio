"""Training loop for FER2013 emotion models."""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.dataset import compute_class_weights, create_dataloaders
from src.model import (
    ARCHITECTURES,
    count_parameters,
    create_model,
    freeze_backbone,
    unfreeze_backbone,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train(
    data_dir: Path,
    output_dir: Path,
    arch: str = "cnn",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 7,
    freeze_epochs: int = 0,
    augment_strength: str = "basic",
    label_smoothing: float = 0.0,
    weight_decay: float = 0.0,
    run_suffix: str = "",
    scheduler_type: str = "plateau",
    resume_path: Path | None = None,
) -> Path:
    device = get_device()
    print(
        f"Using device: {device}, arch: {arch}, "
        f"augment: {augment_strength}, label_smoothing: {label_smoothing}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=0,
        augment_strength=augment_strength,
    )

    model = create_model(arch).to(device)
    start_epoch = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resumed from {resume_path} (epoch {start_epoch})")

    if freeze_epochs > 0:
        freeze_backbone(model)
        print(f"Backbone frozen for first {freeze_epochs} epochs")

    print(f"Trainable parameters: {count_parameters(model):,}")

    class_weights = compute_class_weights(data_dir / "fer2013_train.csv").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    trainable = filter(lambda p: p.requires_grad, model.parameters())
    if weight_decay > 0:
        optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = Adam(trainable, lr=lr)

    if scheduler_type == "cosine":
        scheduler: ReduceLROnPlateau | CosineAnnealingLR = CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    name = f"{arch}{run_suffix}"
    log_path = output_dir / f"training_log_{name}.csv"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint = output_dir / f"best_{name}.pt"

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
        )
        writer.writeheader()

        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                unfreeze_backbone(model)
                optimizer = AdamW(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
                if scheduler_type == "cosine":
                    scheduler = CosineAnnealingLR(
                        optimizer, T_max=epochs - epoch + 1, eta_min=1e-6
                    )
                else:
                    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
                print(f"Epoch {epoch}: backbone unfrozen, lr={lr * 0.1:.1e}")

            t0 = time.perf_counter()
            train_loss, train_acc = run_epoch(
                model, loaders["train"], criterion, optimizer, device, train=True
            )
            val_loss, val_acc = run_epoch(
                model, loaders["val"], criterion, None, device, train=False
            )
            if scheduler_type == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                    "lr": f"{current_lr:.8f}",
                }
            )
            f.flush()

            elapsed = time.perf_counter() - t0
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"lr {current_lr:.2e} | {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "arch": arch,
                        "augment_strength": augment_strength,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    best_checkpoint,
                )
                print(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                    break

    del loaders, model, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    return best_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train emotion model on FER2013")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--arch", choices=ARCHITECTURES, default="cnn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=None,
        help="Freeze MobileNet backbone for N epochs (default: 5 for mobilenet_v3)",
    )
    parser.add_argument(
        "--augment",
        choices=("basic", "strong"),
        default="basic",
        help="Training augmentation strength",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--run-suffix", type=str, default="")
    parser.add_argument(
        "--scheduler",
        choices=("plateau", "cosine"),
        default="plateau",
        help="LR scheduler type",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to fine-tune from")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 1e-4 if args.arch == "mobilenet_v3" else 1e-3
    if args.freeze_epochs is None:
        args.freeze_epochs = 5 if args.arch == "mobilenet_v3" else 0

    # sensible defaults for strong augmentation run
    if args.augment == "strong" and args.arch == "cnn" and args.label_smoothing == 0.0:
        args.label_smoothing = 0.1
    if args.augment == "strong" and args.weight_decay == 0.0:
        args.weight_decay = 1e-4
    if args.augment == "strong" and not args.run_suffix:
        args.run_suffix = "_aug"

    if args.resume and args.lr is None:
        args.lr = 1e-4

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        freeze_epochs=args.freeze_epochs,
        augment_strength=args.augment,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        run_suffix=args.run_suffix,
        scheduler_type=args.scheduler,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
