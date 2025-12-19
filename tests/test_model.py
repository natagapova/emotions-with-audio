"""Tests for EmotionCNN model."""

import torch
import pytest

from src.dataset import NUM_CLASSES
from src.model import EmotionCNN, count_parameters


@pytest.fixture
def model() -> EmotionCNN:
    return EmotionCNN()


def test_output_shape_batch(model: EmotionCNN) -> None:
    x = torch.randn(8, 1, 48, 48)
    out = model(x)
    assert out.shape == (8, NUM_CLASSES)


def test_output_shape_single(model: EmotionCNN) -> None:
    x = torch.randn(1, 1, 48, 48)
    out = model(x)
    assert out.shape == (1, NUM_CLASSES)


def test_forward_no_nan(model: EmotionCNN) -> None:
    x = torch.randn(4, 1, 48, 48)
    out = model(x)
    assert not torch.isnan(out).any()


def test_parameter_count(model: EmotionCNN) -> None:
    params = count_parameters(model)
    assert 1_000_000 < params < 5_000_000
