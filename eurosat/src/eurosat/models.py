"""Network definitions: the shallow two-layer CNN, ResNet-50 and GoogLeNet."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import GoogLeNet_Weights, ResNet50_Weights, googlenet, resnet50

from . import config as C

ARCHS = ("shallow", "resnet50", "googlenet")


class ShallowCNN(nn.Module):
    """The paper's "CNN (two layers)": two convolution + pooling blocks and a linear classifier.

    The paper gives no further detail, so this is our reading of it: no batch normalisation and
    no extra hidden layer, in the spirit of a 2017 "shallow" baseline.  Input 3x64x64.
        conv 5x5, 32 -> ReLU -> maxpool 2   (32 x 32 x 32)
        conv 5x5, 64 -> ReLU -> maxpool 2   (64 x 16 x 16)
        dropout -> linear (16384 -> 10)
    """

    def __init__(self, num_classes: int = C.N_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(64 * 16 * 16, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(arch: str, pretrained: bool, num_classes: int = C.N_CLASSES) -> nn.Module:
    """Create a network.  ``pretrained`` loads torchvision's ImageNet-1k (ILSVRC-2012) weights and
    replaces the 1000-way classifier by a freshly initialised ``num_classes``-way one."""
    if arch == "shallow":
        if pretrained:
            raise ValueError("the shallow CNN has no pretrained weights")
        return ShallowCNN(num_classes)
    torch.hub.set_dir(str(C.DATA_DIR / "pretrained"))   # keep downloaded weights inside the project
    if arch == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "googlenet":
        if pretrained:
            # torchvision drops the two auxiliary heads and sets transform_input=True so that the
            # ported weights receive the input scaling they were trained with.
            m = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
        # From scratch we keep the auxiliary classifiers of the original design (Szegedy et al.).
        return googlenet(weights=None, num_classes=num_classes, aux_logits=True, init_weights=True)
    raise ValueError(f"unknown architecture {arch!r}; choose from {ARCHS}")


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters of the last (classification) layer."""
    if isinstance(model, ShallowCNN):
        return list(model.classifier.parameters())
    return list(model.fc.parameters())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
