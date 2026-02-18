# models/run_sj_dml.py
import torch
import torch.nn as nn
from copy import deepcopy

import torch_directml

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based import neuron, layer, functional


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        cnf=None,
        spiking_neuron=None,
        bn_momentum=0.1,
        **kwargs
    ):
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
    def __init__(
        self,
        block,
        num_block,
        num_classes=10,
        cnf: str = None,
        spiking_neuron: callable = None,
        bn_momentum=0.1,
        **kwargs
    ):
        super().__init__()
        self.in_channels = 8

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum=bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
        )

        self.layer2 = self._make_layer(block, 8,  num_block[0], stride=1, cnf=cnf,
                                       spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 16, num_block[1], stride=2, cnf=cnf,
                                       spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer4 = self._make_layer(block, 32, num_block[2], stride=2, cnf=cnf,
                                       spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)

        self.avgpool = layer.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(32 * 4 * 4, 1024, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(1024, 1024, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(1024, num_classes, bias=False),
        )

    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str = None,
                    spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_channels, out_channels, st, cnf=cnf,
                                spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
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


def ensure_time_batch_first(x: torch.Tensor, T: int) -> torch.Tensor:
    # Debug once if you want:
    # print("ensure_time_batch_first got:", x.dim(), tuple(x.shape))

    if x.dim() == 5:
        # [T,B,C,H,W] or [B,T,C,H,W]
        if x.shape[0] == T:
            return x
        if x.shape[1] == T:
            return x.permute(1, 0, 2, 3, 4).contiguous()
        # If T doesn't match any dimension, assume it's already [T,B,C,H,W]
        return x

    if x.dim() == 4:
        return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    raise RuntimeError(f"Unexpected x.dim={x.dim()} with shape {tuple(x.shape)}")



def main():
    device = torch_directml.device()
    print("Using DirectML device:", device)

    # ===== FashionMNIST transforms =====
    # FashionMNIST is grayscale; model expects 3 channels, so replicate to 3 channels.
    tfm_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    tfm_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm_train)
    test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm_test)

    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # ===== Spiking neuron setup (surrogate gradient is key) =====
    spk = neuron.LIFNode
    spk_kwargs = dict(
        tau=2.0,
        v_threshold=0.5,
        v_reset=0.0,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )

    model = KS32_FullySpiking_Small(
        block=BasicBlock,
        num_block=[5, 5, 5],
        num_classes=10,
        spiking_neuron=spk,
        **spk_kwargs
    ).to(device)

    # ===== Multi-step setting =====
    T = 8
    functional.set_step_mode(model, 'm')

    # ===== Optimizer (DirectML-friendly) =====
    opt = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    # Optional scheduler
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    crit = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # ----- train -----
        model.train()
        correct = total = 0

        for i, (x, y) in enumerate(train_ld):
            x = x.to(device)
            y = y.to(device)

            functional.reset_net(model)
            if epoch == 1 and i == 0:
                print("BATCH x.dim/x.shape:", x.dim(), tuple(x.shape))
            x_seq = ensure_time_batch_first(x, T)   # [T,B,C,H,W]
            out_seq = model(x_seq)                  # [T,B,10]
            out = out_seq.mean(0)                   # [B,10]

            loss = crit(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()

            if i % 100 == 0:
                print(f"Epoch {epoch} batch {i}/{len(train_ld)} loss {loss.item():.4f}")

        train_acc = correct / total

        # ----- eval -----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_ld:
                x = x.to(device)
                y = y.to(device)

                functional.reset_net(model)

                x_seq = ensure_time_batch_first(x, T)
                out_seq = model(x_seq)
                out = out_seq.mean(0)

                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()

        test_acc = correct / total
        print(f"Epoch {epoch} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

        scheduler.step()


if __name__ == "__main__":
    main()

