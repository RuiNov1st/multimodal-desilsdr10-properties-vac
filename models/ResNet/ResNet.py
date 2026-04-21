"""
ResNet-based image encoders for galaxy imaging data.

This module adapts standard ResNet architectures (from torchvision)
for representation learning on multi-band astronomical images.

Key modifications:
- Flexible input channels (to support multi-band images, e.g., g/r/i/z)
- Replacement of the final classification layer with a feature projection layer
- Optional use of pretrained weights (ImageNet)

The output is a latent feature vector used for:
- downstream regression (image-only model)
- multimodal fusion with catalog features

Supported variants:
- ResNet18 / ResNet34 / ResNet50 / ResNet101

Last updated: 2026-04-20
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights
)


# ------------------------------------------------------------
# ResNet50
# ------------------------------------------------------------
class ResNet50(nn.Module):
    """
    ResNet50-based image encoder.

    Modifications:
    - First convolution adapted to arbitrary input channels
    - Final FC layer outputs feature embedding instead of class logits
    """

    def __init__(self, in_channels, output_dim, pretrained=False):
        super(ResNet50, self).__init__()

        # Load backbone
        if pretrained:
            self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.resnet50 = models.resnet50(weights=None)

        # Adapt input channels (for multi-band images)
        self.resnet50.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace classification head with feature projection
        self.resnet50.fc = nn.Linear(
            in_features=2048,
            out_features=output_dim
        )

    def forward(self, x):
        return self.resnet50(x)


# ------------------------------------------------------------
# ResNet34
# ------------------------------------------------------------
class ResNet34(nn.Module):
    """
    ResNet34-based image encoder.
    """

    def __init__(self, in_channels, output_dim, pretrained=False):
        super(ResNet34, self).__init__()

        if pretrained:
            self.resnet34 = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            self.resnet34 = models.resnet34(weights=None)

        self.resnet34.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.resnet34.fc = nn.Linear(
            in_features=512,
            out_features=output_dim
        )

    def forward(self, x):
        return self.resnet34(x)


# ------------------------------------------------------------
# ResNet18
# ------------------------------------------------------------
class ResNet18(nn.Module):
    """
    ResNet18-based image encoder.
    """

    def __init__(self, in_channels, output_dim, pretrained=False):
        super(ResNet18, self).__init__()

        if pretrained:
            self.resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet18 = models.resnet18(weights=None)

        self.resnet18.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.resnet18.fc = nn.Linear(
            in_features=512,
            out_features=output_dim
        )

    def forward(self, x):
        return self.resnet18(x)


# ------------------------------------------------------------
# ResNet101
# ------------------------------------------------------------
class ResNet101(nn.Module):
    """
    ResNet101-based image encoder.
    """

    def __init__(self, in_channels, output_dim, pretrained=False):
        super(ResNet101, self).__init__()

        if pretrained:
            self.resnet101 = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        else:
            self.resnet101 = models.resnet101(weights=None)

        self.resnet101.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.resnet101.fc = nn.Linear(
            in_features=2048,
            out_features=output_dim
        )

    def forward(self, x):
        return self.resnet101(x)


# ------------------------------------------------------------
# Debug / standalone test
# ------------------------------------------------------------
if __name__ == '__main__':
    model = ResNet101(in_channels=7, output_dim=1024)

    x = torch.rand(5, 7, 64, 64)
    out = model(x)

    print(model)
    print(out.shape)