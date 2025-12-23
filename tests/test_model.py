"""Tests for emotion model architectures."""

import torch
import pytest

from src.dataset import NUM_CLASSES
from src.model import EmotionCNN, EmotionMobileNetV3, count_parameters, create_model


@pytest.mark.parametrize("arch", ["cnn", "mobilenet_v3"])
def test_output_shape_batch(arch: str) -> None:
    model = create_model(arch)
    x = torch.randn(8, 1, 48, 48)
    out = model(x)
    assert out.shape == (8, NUM_CLASSES)


@pytest.mark.parametrize("arch", ["cnn", "mobilenet_v3"])
def test_output_shape_single(arch: str) -> None:
    model = create_model(arch)
    x = torch.randn(1, 1, 48, 48)
    out = model(x)
    assert out.shape == (1, NUM_CLASSES)


def test_cnn_parameter_count() -> None:
    params = count_parameters(EmotionCNN())
    assert 1_000_000 < params < 5_000_000


def test_mobilenet_parameter_count() -> None:
    params = count_parameters(EmotionMobileNetV3())
    assert 1_000_000 < params < 3_000_000
