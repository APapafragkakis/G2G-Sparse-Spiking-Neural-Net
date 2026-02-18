from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegreeMaskedConv2d(nn.Module):
    """
    Wrap an existing nn.Conv2d and apply a channel-wise mask to control connectivity.

    Interpretation:
      - each output channel (filter) is a "neuron"
      - we control its fan-in: how many input channels connect to it

    Mask is defined over channel connections only (broadcasts over kH,kW):
      conv.weight: [C_out, C_in_per_group, kH, kW]
      mask      : [C_out, C_in_per_group, 1, 1]

    degree_mode:
      - "exact": enforce exactly K active input channels per output channel
      - "max"  : enforce at most K active input channels per output channel
    """

    def __init__(self, conv: nn.Conv2d, k_in: int, degree_mode: str = "exact"):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(f"conv must be nn.Conv2d, got {type(conv)}")
        if degree_mode not in {"exact", "max"}:
            raise ValueError("degree_mode must be 'exact' or 'max'")
        if k_in <= 0:
            raise ValueError("k_in must be positive")

        self.conv = conv
        self.k_in = int(k_in)
        self.degree_mode = degree_mode

        # Note: for groups>1, conv.weight's 2nd dim is (C_in / groups)
        C_out = conv.weight.shape[0]
        C_in_per = conv.weight.shape[1]

        self.register_buffer("mask", torch.zeros(C_out, C_in_per, 1, 1))
        self.reset_mask()

    @torch.no_grad()
    def reset_mask(self, seed: int | None = None):
        """
        Initialize a random binary mask with exactly K active input channels per output channel.
        If seed is given, mask becomes reproducible.
        """
        self.mask.zero_()
        device = self.mask.device

        g = None
        if seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(int(seed))

        C_out = self.mask.shape[0]
        C_in_per = self.mask.shape[1]
        K = min(self.k_in, C_in_per)

        for o in range(C_out):
            if g is None:
                idx = torch.randperm(C_in_per, device=device)[:K]
            else:
                idx = torch.randperm(C_in_per, generator=g, device=device)[:K]
            self.mask[o, idx, 0, 0] = 1.0

    @torch.no_grad()
    def project_mask_(self):
        """
        Enforce the degree constraint on the current mask.

        - For "exact": row-sum == K (drop extras, add missing)
        - For "max"  : row-sum <= K (drop extras only)

        Uses random add/drop by default (simple + fast).
        """
        device = self.mask.device
        C_out = self.mask.shape[0]
        C_in_per = self.mask.shape[1]
        K = min(self.k_in, C_in_per)

        # ensure binary
        self.mask.clamp_(0, 1)
        self.mask.round_()

        for o in range(C_out):
            row = self.mask[o, :, 0, 0]
            ones = int(row.sum().item())

            if ones > K:
                on_idx = torch.where(row > 0.5)[0]
                perm = torch.randperm(on_idx.numel(), device=device)
                drop = on_idx[perm[: (ones - K)]]
                row[drop] = 0.0

            if self.degree_mode == "exact" and ones < K:
                off_idx = torch.where(row < 0.5)[0]
                if off_idx.numel() > 0:
                    perm = torch.randperm(off_idx.numel(), device=device)
                    add = off_idx[perm[: (K - ones)]]
                    row[add] = 1.0

            self.mask[o, :, 0, 0] = row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.conv.weight * self.mask  # broadcast over kH,kW
        return F.conv2d(
            x,
            w_eff,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


def replace_convs_with_degree_mask(module: nn.Module, k_in: int, degree_mode: str = "exact"):
    """
    Recursively replace nn.Conv2d modules with DegreeMaskedConv2d wrappers.

    This keeps original conv weights/bias/stride/padding/etc.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, DegreeMaskedConv2d(child, k_in=k_in, degree_mode=degree_mode))
        else:
            replace_convs_with_degree_mask(child, k_in=k_in, degree_mode=degree_mode)


@torch.no_grad()
def reset_all_degree_masks(model: nn.Module, seed: int | None = None):
    """Reset all DegreeMaskedConv2d masks (optionally reproducibly)."""
    for m in model.modules():
        if isinstance(m, DegreeMaskedConv2d):
            m.reset_mask(seed=seed)


@torch.no_grad()
def project_all_degree_masks(model: nn.Module):
    """Project all DegreeMaskedConv2d masks back to the degree constraint."""
    for m in model.modules():
        if isinstance(m, DegreeMaskedConv2d):
            m.project_mask_()
