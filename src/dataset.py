"""FER2013 dataset loading and preprocessing."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

NUM_CLASSES = len(EMOTION_LABELS)
IMAGE_SIZE = 48


def _csv_to_npz(csv_path: Path, npz_path: Path) -> None:
    """Convert CSV split to compressed NPZ for memory-mapped loading."""
    with csv_path.open(newline="") as f:
        row_count = sum(1 for _ in f) - 1

    pixels = np.empty((row_count, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    emotions = np.empty(row_count, dtype=np.int64)

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pixels[i] = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8).reshape(
                IMAGE_SIZE, IMAGE_SIZE
            )
            emotions[i] = int(row["emotion"])

    np.savez_compressed(npz_path, pixels=pixels, emotions=emotions)


def _ensure_npz(csv_path: Path) -> Path:
    npz_path = csv_path.with_suffix(".npz")
    if not npz_path.exists() or npz_path.stat().st_mtime < csv_path.stat().st_mtime:
        _csv_to_npz(csv_path, npz_path)
    return npz_path


def _build_transforms(augment: bool) -> transforms.Compose:
    ops: list = []
    if augment:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transforms.Compose(ops)


class FER2013Dataset(Dataset):
    """48x48 grayscale facial emotion images from FER2013 CSV splits."""

    def __init__(
        self,
        csv_path: str | Path,
        augment: bool = False,
    ) -> None:
        csv_path = Path(csv_path)
        npz_path = _ensure_npz(csv_path)
        data = np.load(npz_path, mmap_mode="r")
        self.pixels = data["pixels"]
        self.emotions = data["emotions"]
        self.transform = _build_transforms(augment)

    def __len__(self) -> int:
        return len(self.emotions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(self.pixels[idx], mode="L")
        image = self.transform(image)
        label = int(self.emotions[idx])
        return image, label


def get_class_counts(csv_path: str | Path) -> np.ndarray:
    """Return per-class sample counts indexed by emotion id."""
    df = pd.read_csv(csv_path, usecols=["emotion"])
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for emotion, count in df["emotion"].value_counts().items():
        counts[int(emotion)] = count
    return counts


def compute_class_weights(csv_path: str | Path) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = get_class_counts(csv_path).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(csv_path: str | Path) -> WeightedRandomSampler:
    """WeightedRandomSampler to balance minority classes during training."""
    counts = get_class_counts(csv_path).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    sample_weights = 1.0 / counts
    emotions = pd.read_csv(csv_path, usecols=["emotion"])["emotion"].values
    weights = sample_weights[emotions]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(emotions),
        replacement=True,
    )


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders with native FER2013 splits."""
    data_dir = Path(data_dir)
    loaders: dict[str, DataLoader] = {}

    train_csv = data_dir / "fer2013_train.csv"
    train_ds = FER2013Dataset(train_csv, augment=True)

    train_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if use_weighted_sampler:
        train_kwargs["sampler"] = make_weighted_sampler(train_csv)
    else:
        train_kwargs["shuffle"] = True

    loaders["train"] = DataLoader(train_ds, **train_kwargs)

    for split in ("val", "test"):
        csv_path = data_dir / f"fer2013_{split}.csv"
        ds = FER2013Dataset(csv_path, augment=False)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


def print_class_distribution(data_dir: str | Path) -> None:
    """Print per-split class distribution."""
    data_dir = Path(data_dir)
    for split in ("train", "val", "test"):
        csv_path = data_dir / f"fer2013_{split}.csv"
        if not csv_path.exists():
            print(f"  {split}: file not found ({csv_path})")
            continue
        counts = get_class_counts(csv_path)
        total = counts.sum()
        print(f"\n{split} ({total} samples):")
        for idx, count in enumerate(counts):
            pct = 100.0 * count / total if total else 0.0
            print(f"  {EMOTION_LABELS[idx]:10s}: {count:5d} ({pct:5.1f}%)")
