# models/frozen_resnet_cut.py
import torch
from torch import nn


class FrozenTruncatedResNet(nn.Module):
    """
    Frozen pretrained CIFAR ResNet-32 from torch.hub, truncated early to produce WEAK features.

    Returns flattened embeddings:
      - layer1: 16 * pool_hw * pool_hw
      - layer2: 32 * pool_hw * pool_hw   
      - layer3: 64 * pool_hw * pool_hw
    """

    def __init__(self, dataset: str = "cifar10", cut_at: str = "layer2", pool_hw: int = 4):
        super().__init__()
        if dataset not in {"cifar10", "cifar100"}:
            raise ValueError(f"dataset must be 'cifar10' or 'cifar100', got {dataset}")
        if cut_at not in {"layer1", "layer2", "layer3"}:
            raise ValueError(f"cut_at must be 'layer1'/'layer2'/'layer3', got {cut_at}")

        base = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            f"{dataset}_resnet32",
            pretrained=True,
        )

        self.stem = nn.Sequential(base.conv1, base.bn1, nn.ReLU(inplace=True))
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        self.cut_at = cut_at
        self.pool_hw = int(pool_hw)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_hw, self.pool_hw))

        # Freeze everything
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @property
    def out_dim(self) -> int:
        ch = {"layer1": 16, "layer2": 32, "layer3": 64}[self.cut_at]
        return ch * self.pool_hw * self.pool_hw

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.stem(x)
        x = self.layer1(x)
        if self.cut_at == "layer1":
            return self.pool(x).flatten(1)

        x = self.layer2(x)
        if self.cut_at == "layer2":
            return self.pool(x).flatten(1)

        x = self.layer3(x)
        return self.pool(x).flatten(1)
