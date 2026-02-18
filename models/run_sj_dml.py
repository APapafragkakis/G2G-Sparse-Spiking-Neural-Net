import torch
import torch.nn as nn
from copy import deepcopy

import torch_directml

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from spikingjelly.activation_based import neuron, layer, functional


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, cnf=None, spiking_neuron=None, bn_momentum=0.1, **kwargs):
        super().__init__()
        self.conv1 = layer.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))

        self.conv2 = layer.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                layer.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_ch, momentum=bn_momentum),
            )

    def forward(self, x):
        identity = x
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        out = self.sn2(out)
        return out


class KS32_FullySpiking_Small(nn.Module):
    def __init__(self, block, num_block, num_classes=10, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super().__init__()
        self.in_channels = 8

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum=bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
        )

        self.layer2 = self._make_layer(block, 8,  num_block[0], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 16, num_block[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer4 = self._make_layer(block, 32, num_block[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)

        self.avgpool = layer.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(32 * 4 * 4, 1024, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(1024, 1024, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(1024, num_classes, bias=False),
        )

    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str=None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_channels, out_channels, st, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


def main():
    device = torch_directml.device()
    print("Using DirectML device:", device)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)

    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_ld  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    spk = neuron.LIFNode

    model = KS32_FullySpiking_Small(
        block=BasicBlock,
        num_block=[5, 5, 5],
        num_classes=10,
        spiking_neuron=spk,
        tau=2.0,
        v_threshold=1.0,
        v_reset=0.0
    ).to(device)

    functional.set_step_mode(model, 's')

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, 6):
        model.train()
        correct = total = 0

        for x, y in train_ld:
            x = x.to(device)
            y = y.to(device)

            functional.reset_net(model)

            out = model(x)
            loss = crit(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()

        train_acc = correct / total

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_ld:
                x = x.to(device)
                y = y.to(device)
                functional.reset_net(model)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()

        test_acc = correct / total
        print(f"Epoch {epoch} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
