"""Lightweight ensemble inference (sequential, low memory)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import create_dataloaders
from src.evaluate import plot_confusion_matrix
from src.model import load_model_from_checkpoint
from src.train import get_device


@torch.no_grad()
def ensemble_predict(
    models: list[nn.Module],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        logits_sum = None

        for model in models:
            out = model(images)
            if tta:
                flipped = torch.flip(images, dims=[-1])
                out = (out + model(flipped)) / 2
            logits_sum = out if logits_sum is None else logits_sum + out

        assert logits_sum is not None
        preds = (logits_sum / len(models)).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble evaluation")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=[Path("models/best_cnn_aug.pt"), Path("models/best_cnn_v2.pt")],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    device = get_device()
    models = [load_model_from_checkpoint(p, device) for p in args.checkpoints]
    loaders = create_dataloaders(
        args.data_dir, batch_size=args.batch_size, use_weighted_sampler=False, num_workers=0
    )

    y_true, y_pred = ensemble_predict(models, loaders["test"], device, tta=args.tta)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Ensemble ({len(models)} models, tta={args.tta})")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Macro F1:      {f1:.4f}")

    out_dir = Path("models/evaluation_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
