# models/cnn_embedder.py
import torch
from torch import nn
import torch.nn.functional as F


class CNNEmbedder1Layer(nn.Module):
    """
    Simple 1-layer CNN embedder:
      Conv -> ReLU -> MaxPool -> Flatten -> Linear (to embedding_dim)
    Works for CIFAR (3x32x32) and FashionMNIST (1x28x28).
    """

    def __init__(
        self,
        in_channels: int,
        image_hw: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        pool: int = 2,
        embedding_dim: int = 256,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(pool)

        # compute flatten dim (assumes square images)
        h2 = image_hw // pool
        w2 = image_hw // pool
        flat_dim = conv_channels * h2 * w2

        self.proj = nn.Linear(flat_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, embedding_dim]
        """
        x = self.pool(F.relu(self.conv(x)))
        x = x.flatten(1)
        x = self.proj(x)
        return x