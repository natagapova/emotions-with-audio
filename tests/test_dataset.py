"""Tests for FER2013 dataset preprocessing."""

from pathlib import Path

import pytest
import torch

from src.dataset import FER2013Dataset, IMAGE_SIZE, NUM_CLASSES


DATA_DIR = Path("data")
TRAIN_CSV = DATA_DIR / "fer2013_train.csv"


@pytest.fixture(scope="module")
def dataset() -> FER2013Dataset:
    if not TRAIN_CSV.exists():
        pytest.skip("FER2013 data not downloaded. Run: python scripts/download_fer2013.py")
    return FER2013Dataset(TRAIN_CSV, augment=False)


def test_dataset_length(dataset: FER2013Dataset) -> None:
    assert len(dataset) == 28709


def test_tensor_shape(dataset: FER2013Dataset) -> None:
    image, label = dataset[0]
    assert image.shape == (1, IMAGE_SIZE, IMAGE_SIZE)


def test_tensor_range(dataset: FER2013Dataset) -> None:
    image, _ = dataset[0]
    assert image.dtype == torch.float32
    # Normalized with mean=0.5, std=0.5 -> range approximately [-1, 1]
    assert image.min() >= -1.0
    assert image.max() <= 1.0


def test_label_range(dataset: FER2013Dataset) -> None:
    _, label = dataset[0]
    assert 0 <= label < NUM_CLASSES
