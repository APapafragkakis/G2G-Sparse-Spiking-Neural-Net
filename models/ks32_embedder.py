import torch
import torch.nn as nn
import torch.nn.functional as F

class KS32GroupedEmbedder(nn.Module):
    """
    1-layer CNN embedder:
    Conv2d(in_ch -> base_ch, 3x3, padding=1) + ReLU + AdaptiveAvgPool(4x4) + Linear -> emb_dim
    Returns: [B, emb_dim]
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, emb_dim: int = 512):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(base_ch * 4 * 4, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
