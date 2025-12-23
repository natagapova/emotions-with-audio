"""Evaluate trained emotion model on test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.dataset import EMOTION_LABELS, create_dataloaders
from src.model import load_model_from_checkpoint
from src.train import get_device


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    return load_model_from_checkpoint(checkpoint_path, device)


def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_LABELS))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(EMOTION_LABELS)),
        yticks=np.arange(len(EMOTION_LABELS)),
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
) -> dict:
    device = get_device()
    model = load_model(checkpoint_path, device)
    loaders = create_dataloaders(data_dir, batch_size=batch_size, use_weighted_sampler=False)

    y_true, y_pred = collect_predictions(model, loaders["test"], device)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTION_LABELS,
        output_dict=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path)

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1-score": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in EMOTION_LABELS
        },
        "confusion_matrix_path": str(cm_path),
    }

    results_path = output_dir / "evaluation_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"\nPer-class metrics:")
    for label in EMOTION_LABELS:
        m = results["per_class"][label]
        print(
            f"  {label:10s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1-score']:.3f}  (n={m['support']})"
        )
    print(f"\nConfusion matrix saved to {cm_path}")
    print(f"Results saved to {results_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate emotion CNN")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_cnn.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
