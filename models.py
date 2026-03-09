from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_V2_S_Weights


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def _resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _efficientnet_v2_s(num_classes: int, pretrained: bool) -> nn.Module:
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    if isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def get_model(name: str, num_classes: int, pretrained: bool = True, **kwargs: Any) -> nn.Module:
    name = name.lower().strip()
    if name == "cnn":
        if pretrained:
            print("Note: pretrained=True is ignored for cnn (random init).")
        return SimpleCNN(num_classes)
    if name == "resnet50":
        return _resnet50(num_classes, pretrained)
    if name == "efficientnet_v2_s":
        return _efficientnet_v2_s(num_classes, pretrained)
    raise ValueError(f"Unknown model name: {name}")
