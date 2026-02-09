# models/ks32_grouped.py
import torch
import torch.nn as nn

class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, cardinality=4, width_per_group=4):
        super().__init__()
        D = cardinality * width_per_group

        self.conv1 = nn.Conv2d(in_ch, D, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(D)

        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2   = nn.BatchNorm2d(D)

        self.conv3 = nn.Conv2d(D, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class KS32GroupedEmbedder(nn.Module):
    """
    CIFAR: input [B,3,32,32]
    Stages: 8 -> 16 -> 32, output embedding dim = 32*4*4 = 512
    """
    def __init__(self, num_blocks=(5,5,5), cardinality=4, width_per_group=4, out_pool_hw=4):
        super().__init__()
        self.in_ch = 8

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_ch),
            nn.ReLU(inplace=True),
        )

        self.layer2 = self._make_layer(8,  num_blocks[0], stride=1, cardinality=cardinality, width_per_group=width_per_group)
        self.layer3 = self._make_layer(16, num_blocks[1], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.layer4 = self._make_layer(32, num_blocks[2], stride=2, cardinality=cardinality, width_per_group=width_per_group)

        self.pool = nn.AdaptiveAvgPool2d((out_pool_hw, out_pool_hw))
        self.out_dim = 32 * out_pool_hw * out_pool_hw  # default 32*4*4=512

        # Kaiming init (συμβατό με ReLU)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, out_ch, n_blocks, stride, cardinality, width_per_group):
        layers = []
        layers.append(ResNeXtBlock(self.in_ch, out_ch, stride=stride,
                                  cardinality=cardinality, width_per_group=width_per_group))
        self.in_ch = out_ch
        for _ in range(n_blocks - 1):
            layers.append(ResNeXtBlock(self.in_ch, out_ch, stride=1,
                                      cardinality=cardinality, width_per_group=width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)  # [B, 512]
