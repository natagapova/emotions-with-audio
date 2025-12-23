"""Model architectures for FER2013 emotion classification."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.dataset import NUM_CLASSES

ARCHITECTURES = ("cnn", "mobilenet_v3")


class EmotionCNN(nn.Module):
    """Small CNN: 4 conv blocks (32→64→128→256) + 2 FC layers."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.arch = "cnn"

        self.features = nn.Sequential(
            self._conv_block(1, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class EmotionMobileNetV3(nn.Module):
    """MobileNetV3-Small with ImageNet pretraining, adapted for 48×48 grayscale."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.arch = "mobilenet_v3"

        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return self.backbone(x)


def create_model(arch: str = "cnn", num_classes: int = NUM_CLASSES) -> nn.Module:
    if arch == "cnn":
        return EmotionCNN(num_classes=num_classes)
    if arch == "mobilenet_v3":
        return EmotionMobileNetV3(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {arch!r}. Choose from {ARCHITECTURES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "cnn")
    model = create_model(arch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def freeze_backbone(model: nn.Module) -> None:
    if isinstance(model, EmotionMobileNetV3):
        for param in model.backbone.features.parameters():
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    if isinstance(model, EmotionMobileNetV3):
        for param in model.backbone.features.parameters():
            param.requires_grad = True
