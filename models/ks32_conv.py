# conv_edge_pruning.py
# Channel-edge pruning for Conv2d layers (edges: input channel c -> output channel c')
#
# Implements:
# 1) Pre-training pruning (random mask at init)
# 2) Dynamic pruning during training (periodic prune lowest-importance active edges + random rewiring)
# 3) Post-training pruning (prune low-importance edges + optional fine-tune outside this module)
#
# Assumptions:
# - Edge importance is L1 norm of the kernel: importance(c->c') = ||K_{c',c}||_1
# - Ablating an edge means removing the entire kernel W[c',c,:,:] via a binary mask
# - Supports standard (dense) conv only: groups == 1

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------
# Masked conv wrapper
# -----------------------------

class ChannelMaskedConv2d(nn.Module):
    """
    Wraps nn.Conv2d and applies a channel-edge mask of shape [C_out, C_in].

    Effective weight:
        W_eff = W * mask[:, :, None, None]

    Notes:
    - Supports groups == 1 only.
    """
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("ChannelMaskedConv2d expects nn.Conv2d")
        if conv.groups != 1:
            raise NotImplementedError("Only groups==1 supported")

        self.conv = conv
        self.register_buffer(
            "mask",
            torch.ones(conv.out_channels, conv.in_channels, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.conv.weight * self.mask[:, :, None, None]
        return nn.functional.conv2d(
            x,
            w_eff,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    def extra_repr(self) -> str:
        active = int(self.mask.sum().item())
        total = self.mask.numel()
        return f"masked_edges={active}/{total} | sparsity={1 - active/total:.2%}"


# -----------------------------
# Utilities
# -----------------------------

def iter_masked_convs(model: nn.Module) -> Iterable[Tuple[str, ChannelMaskedConv2d]]:
    for name, m in model.named_modules():
        if isinstance(m, ChannelMaskedConv2d):
            yield name, m


def wrap_conv2d_with_edge_masks_(
    model: nn.Module,
    *,
    include_names: Optional[List[str]] = None
) -> nn.Module:
    """
    In-place replacement of nn.Conv2d with ChannelMaskedConv2d.
    """
    def should_wrap(full_name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Conv2d):
            return False
        if module.groups != 1:
            return False
        if include_names is None:
            return True
        return any(s in full_name for s in include_names)

    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}".lstrip(".")
            if should_wrap(full, child):
                setattr(parent, child_name, ChannelMaskedConv2d(child))
    return model


# -----------------------------
# Core pruning logic
# -----------------------------

@torch.no_grad()
def init_random_sparsity_(layer: ChannelMaskedConv2d, sparsity: float) -> None:
    """
    Pre-training pruning: randomly deactivate a fixed percentage of edges.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0,1)")

    total = layer.mask.numel()
    num_prune = int(round(sparsity * total))

    layer.mask.fill_(1.0)
    if num_prune == 0:
        return

    perm = torch.randperm(total, device=layer.mask.device)
    layer.mask.view(-1)[perm[:num_prune]] = 0.0


@torch.no_grad()
def edge_importance_l1(layer: ChannelMaskedConv2d) -> torch.Tensor:
    """
    importance(c -> c') = ||K_{c',c}||_1
    """
    return layer.weight.abs().sum(dim=(2, 3))


@torch.no_grad()
def prune_low_importance_active_(layer: ChannelMaskedConv2d, prune_frac: float) -> int:
    """
    Prune lowest-importance ACTIVE edges.
    """
    if prune_frac <= 0:
        return 0

    mask = layer.mask
    active = mask.bool()
    n_active = int(active.sum().item())
    if n_active == 0:
        return 0

    n_prune = int(prune_frac * n_active)
    if n_prune < 1:
        return 0

    imp = edge_importance_l1(layer)
    active_imp = imp[active].view(-1)

    k = min(n_prune, active_imp.numel())
    threshold, _ = torch.kthvalue(active_imp, k)

    to_prune = (imp <= threshold) & active
    pruned = int(to_prune.sum().item())
    mask[to_prune] = 0.0
    return pruned


@torch.no_grad()
def rewire_random_inactive_(layer: ChannelMaskedConv2d, num_to_grow: int) -> int:
    """
    Randomly activate inactive edges.
    """
    if num_to_grow <= 0:
        return 0

    inactive_idx = (~layer.mask.bool()).nonzero(as_tuple=False)
    if inactive_idx.numel() == 0:
        return 0

    n = min(num_to_grow, inactive_idx.size(0))
    perm = torch.randperm(inactive_idx.size(0), device=layer.mask.device)[:n]
    grow = inactive_idx[perm]

    layer.mask[grow[:, 0], grow[:, 1]] = 1.0
    return int(n)


@torch.no_grad()
def dynamic_prune_and_rewire_(layer: ChannelMaskedConv2d, prune_frac: float) -> Dict[str, int]:
    """
    Prune + rewire while keeping sparsity constant.
    """
    pruned = prune_low_importance_active_(layer, prune_frac)
    grown = rewire_random_inactive_(layer, pruned)
    return {"pruned": pruned, "grown": grown}


# -----------------------------
# Model-level APIs
# -----------------------------

@dataclass
class SparsityStats:
    total_edges: int
    active_edges: int

    @property
    def sparsity(self) -> float:
        return 1.0 - (self.active_edges / self.total_edges)


@torch.no_grad()
def compute_model_sparsity(model: nn.Module) -> SparsityStats:
    total, active = 0, 0
    for _, layer in iter_masked_convs(model):
        total += layer.mask.numel()
        active += int(layer.mask.sum().item())
    return SparsityStats(total, active)


@torch.no_grad()
def pre_training_prune_model_(model: nn.Module, sparsity: float) -> None:
    for _, layer in iter_masked_convs(model):
        init_random_sparsity_(layer, sparsity)


@torch.no_grad()
def dynamic_pruning_step_model_(model: nn.Module, prune_frac: float) -> Dict[str, Dict[str, int]]:
    out = {}
    for name, layer in iter_masked_convs(model):
        out[name] = dynamic_prune_and_rewire_(layer, prune_frac)
    return out


@torch.no_grad()
def post_training_prune_model_(model: nn.Module, prune_frac_of_active: float) -> Dict[str, int]:
    out = {}
    for name, layer in iter_masked_convs(model):
        out[name] = prune_low_importance_active_(layer, prune_frac_of_active)
    return out


def format_sparsity(stats: SparsityStats) -> str:
    return f"active_edges={stats.active_edges}/{stats.total_edges} | sparsity={stats.sparsity:.2%}"
