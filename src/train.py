import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn
from models.dense_snn import DenseSNN
from models.index_snn import IndexSNN, IndexSparseLinear
from models.random_snn import RandomSNN, RandomGroupSparseLinear
from models.mixer_snn import MixerSNN, MixerSparseLinear
from models.ks32_grouped import KS32GroupedEmbedder

from data.fashionmnist import get_fashion_loaders
from data.cifar10_100 import get_cifar10_loaders, get_cifar100_loaders
from utils.encoding import encode_input
from snntorch import spikegen
import snntorch.functional as SF
import warnings

warnings.filterwarnings("ignore", message=".*aten::lerp.Scalar_out.*")

SPARSE_LAYER_TYPES = (MixerSparseLinear, IndexSparseLinear, RandomGroupSparseLinear)
SPARSE_MODEL_TYPES = (MixerSNN, IndexSNN, RandomSNN)

try:
    import torch_directml
    has_dml = True
except ImportError:
    has_dml = False


def select_device():
    if has_dml:
        device = torch_directml.device()
        print(f"Using DirectML device: {device}")
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
        return device
    print("No GPU backend available — using CPU.")
    return torch.device("cpu")


batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 2048
num_classes = 10
num_epochs = 20
lr = 1e-3

enc_mode = "current"
enc_scale = 1.0
enc_bias = 0.0

global_step = 0
UPDATE_INTERVAL = 1000
cp_mode = "set"
cg_mode = "hebb"

hebb_buffer = {"fc1": None, "fc2": None, "fc3": None}

IS_TTY = sys.stdout.isatty()


def firing_rate_loss(spk_stacks, target=0.09, T_steps=50):
    loss = 0.0
    for spk in spk_stacks:
        if spk.dim() == 3:
            rates_per_neuron = spk.mean(dim=(0, 1))
        elif spk.dim() == 2:
            rates_per_neuron = spk.mean(dim=0) / float(T_steps)
        else:
            raise ValueError(f"Unexpected spk shape: {spk.shape}")
        loss += (rates_per_neuron - target).pow(2).sum()
    return loss


def get_progressive_params(epoch, num_epochs, warmup_epochs=10):
    if epoch <= warmup_epochs:
        return 0.0, 0.25
    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
    lambda_start = 1e-4
    lambda_max = 1e-2
    lambda_coef = lambda_start + progress * (lambda_max - lambda_start)
    target_start = 0.25
    target_final = 0.09
    target_rate = target_start - progress * (target_start - target_final)
    return lambda_coef, target_rate


def get_checkpoint_path(args):
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    p_str = f"{args.p_inter:.2f}".replace(".", "_")
    
    # Updated prefix logic
    if args.use_cnn_embed:
        prefix = f"cnn_{args.dataset}"
    elif args.use_resnet:
        prefix = f"resnet_{args.dataset}"
    else:
        prefix = args.dataset
    
    if args.sparsity_mode == "dynamic":
        filename = f"{prefix}_{args.model}_p{p_str}_T{args.T}_{args.enc}_cp{args.cp}_cg{args.cg}.pth"
    else:
        filename = f"{prefix}_{args.model}_p{p_str}_T{args.T}_{args.enc}.pth"
    return os.path.join(checkpoint_dir, filename)


def get_done_marker_path(args):
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    p_str = f"{args.p_inter:.2f}".replace(".", "_")
    
    # Updated prefix logic
    if args.use_cnn_embed:
        prefix = f"cnn_{args.dataset}"
    elif args.use_resnet:
        prefix = f"resnet_{args.dataset}"
    else:
        prefix = args.dataset
    
    if args.sparsity_mode == "dynamic":
        filename = f"{prefix}_{args.model}_p{p_str}_T{args.T}_{args.enc}_cp{args.cp}_cg{args.cg}.DONE"
    else:
        filename = f"{prefix}_{args.model}_p{p_str}_T{args.T}_{args.enc}.DONE"
    return os.path.join(checkpoint_dir, filename)


def save_checkpoint(epoch, model, optimizer, args, metrics=None):
    checkpoint_path = get_checkpoint_path(args)
    state_dict = model.state_dict()
    
    # Remove frozen feature extractor weights from checkpoint (if using pretrained ResNet)
    if getattr(model, "use_resnet", False) and not getattr(model, "use_cnn_embed", False):
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("feature_extractor.")}
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'global_step': global_step,
    }
    if metrics is not None:
        checkpoint['metrics'] = metrics
    torch.save(checkpoint, checkpoint_path)
    print(f"[Checkpoint saved: epoch {epoch}]")


def load_checkpoint(model, optimizer, args):
    checkpoint_path = get_checkpoint_path(args)
    if not os.path.exists(checkpoint_path):
        return 1
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global global_step
        global_step = checkpoint.get('global_step', 0)
        start_epoch = checkpoint['epoch'] + 1
        print(f"[Checkpoint loaded: resuming from epoch {start_epoch}]")
        return start_epoch
    except Exception as e:
        print(f"[Warning: Failed to load checkpoint: {e}]")
        return 1


def build_model(model_name, p_inter, dataset, use_resnet, args):
    feature_extractor = None
    feature_dim = input_dim

    if use_resnet and dataset in ["cifar10", "cifar100"]:
        cut_at = "layer1"
        pool_hw = 4  # 32 * 4 * 4 = 512

        print(f"Loading FrozenTruncatedResNet (pretrained) for {dataset} | cut_at={cut_at} | pool_hw={pool_hw} ...")
        feature_extractor = FrozenTruncatedResNet(
        dataset=dataset,
        cut_at=cut_at,
        pool_hw=pool_hw,
        cardinality=args.resnet_cardinality,
        width_per_group=args.resnet_width_per_group,
    )
        feature_dim = feature_extractor.out_dim  # 512
        print(f"FrozenTruncatedResNet ready. feature_dim={feature_dim}")

    # --- Build SNN classifier ---
    if model_name == "dense":
        model = DenseSNN(feature_dim, hidden_dim, num_classes)
    elif model_name == "index":
        model = IndexSNN(feature_dim, hidden_dim, num_classes, 8, 1.0, p_inter)
    elif model_name == "random":
        model = RandomSNN(feature_dim, hidden_dim, num_classes, 8, 1.0, p_inter)
    elif model_name == "mixer":
        model = MixerSNN(feature_dim, hidden_dim, num_classes, 8, 1.0, p_inter)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Attach feature extractor
    model.feature_extractor = feature_extractor

    # This flag is used by _make_input_sequence to decide no_grad + spikegen.rate
    model.use_resnet = bool(use_resnet and feature_extractor is not None)

    return model

def update_hebb_buffer(input_spikes, activations):
    global hebb_buffer
    hebb_buffer["fc1"] = {"pre": input_spikes.cpu(), "post": activations["layer1"].cpu()}
    hebb_buffer["fc2"] = {"pre": activations["layer1"].cpu(), "post": activations["layer2"].cpu()}
    hebb_buffer["fc3"] = {"pre": activations["layer2"].cpu(), "post": activations["layer3"].cpu()}


def compute_ch_matrix(pre_batch, post_batch):
    T_steps, B, N_in = pre_batch.shape
    _, _, N_out = post_batch.shape
    pre_flat = pre_batch.reshape(T_steps * B, N_in).t()
    post_flat = post_batch.reshape(T_steps * B, N_out).t()
    eps = 1e-8
    pre_norm = pre_flat / (pre_flat.norm(dim=1, keepdim=True) + eps)
    post_norm = post_flat / (post_flat.norm(dim=1, keepdim=True) + eps)
    return post_norm @ pre_norm.t()


def _grow_connections(mask_cpu, ch_cpu, num_to_grow, mode):
    inactive = ~mask_cpu.bool()
    inactive_idx = inactive.nonzero(as_tuple=False)
    if inactive_idx.size(0) == 0 or num_to_grow <= 0:
        return mask_cpu, 0
    num_to_grow = min(num_to_grow, inactive_idx.size(0))
    if mode == "hebb":
        scores = ch_cpu[inactive]
        _, top_idx = torch.topk(scores, k=num_to_grow)
        grow_idx = inactive_idx[top_idx]
    else:
        perm = torch.randperm(inactive_idx.size(0))[:num_to_grow]
        grow_idx = inactive_idx[perm]
    mask_cpu[grow_idx[:, 0], grow_idx[:, 1]] = 1
    return mask_cpu, num_to_grow


def dst_update_layer_cp_cg_single(layer, layer_name, prune_frac, cp_mode_local, cg_mode_local):
    global hebb_buffer
    buf = hebb_buffer.get(layer_name)
    if buf is None:
        return
    pre_batch = buf["pre"]
    post_batch = buf["post"]
    weight = layer.weight.data
    mask = layer.mask
    device = weight.device
    mask_cpu = mask.cpu()
    w_cpu = weight.cpu()
    active = mask_cpu.bool()
    num_active = active.sum().item()
    if num_active == 0:
        return
    total_to_prune = int(prune_frac * num_active)
    if total_to_prune < 1:
        return
    ch_cpu = None
    if cp_mode_local == "hebb" or cg_mode_local == "hebb":
        ch_cpu = compute_ch_matrix(pre_batch, post_batch)
    num_pruned = 0
    if cp_mode_local == "set":
        active_weights = w_cpu[active].abs().view(-1)
        n_eff = min(total_to_prune, active_weights.numel())
        if n_eff > 0:
            threshold, _ = torch.kthvalue(active_weights, n_eff)
            prune_mask = (w_cpu.abs() <= threshold) & active
            mask_cpu[prune_mask] = 0
            num_pruned = prune_mask.sum().item()
    elif cp_mode_local == "random":
        active_idx = active.nonzero(as_tuple=False)
        n_eff = min(total_to_prune, active_idx.size(0))
        if n_eff > 0:
            perm = torch.randperm(active_idx.size(0))[:n_eff]
            prune_idx = active_idx[perm]
            mask_cpu[prune_idx[:, 0], prune_idx[:, 1]] = 0
            num_pruned = n_eff
    elif cp_mode_local == "hebb":
        active_idx = active.nonzero(as_tuple=False)
        n_eff = min(total_to_prune, active_idx.size(0))
        if n_eff > 0:
            scores = ch_cpu[active]
            _, bottom_idx = torch.topk(scores, k=n_eff, largest=False)
            prune_idx = active_idx[bottom_idx]
            mask_cpu[prune_idx[:, 0], prune_idx[:, 1]] = 0
            num_pruned = n_eff
    mask_cpu, num_grown = _grow_connections(mask_cpu, ch_cpu, num_pruned, cg_mode_local)
    layer.mask.copy_(mask_cpu.to(device))


def dst_step(model):
    global cp_mode, cg_mode
    prune_frac = 0.1
    for layer_name, layer_obj in [("fc1", model.fc1), ("fc2", model.fc2), ("fc3", model.fc3)]:
        if isinstance(layer_obj, SPARSE_LAYER_TYPES):
            dst_update_layer_cp_cg_single(layer_obj, layer_name, prune_frac, cp_mode, cg_mode)


def _minmax_per_sample(feat, eps=1e-8):
    """Normalize features per sample to [0, 1] range."""
    mn = feat.min(dim=1, keepdim=True).values
    mx = feat.max(dim=1, keepdim=True).values
    return (feat - mn) / (mx - mn + eps)


def _make_input_sequence(images, device, model, return_embeddings: bool = False):
    """
    Convert images to spike-encoded input sequence.
    
    Args:
        images: Input images [B, C, H, W] or [B, H*W]
        device: Target device
        model: Model with optional feature_extractor
        return_embeddings: If True, also return embeddings [B, D]
    
    Returns:
        x_seq: Spike-encoded sequence [T, B, feature_dim]
        emb (optional): Embeddings [B, feature_dim] if return_embeddings=True
    """
    # If we have any feature extractor (ResNet or CNN embedder), use it
    if getattr(model, "feature_extractor", None) is not None:
        images = images.to(device, non_blocking=True)

        # Use torch.no_grad() only for frozen ResNet, enable grad for trainable CNN
        if getattr(model, "use_resnet", False):
            with torch.no_grad():
                emb = model.feature_extractor(images)
        else:
            # Trainable CNN embedder - keep gradients
            emb = model.feature_extractor(images)

        # Ensure emb is [B, D]
        if emb.dim() > 2:
            emb = emb.flatten(1)

        # Keep raw embeddings for analysis (t-SNE/UMAP/etc.)
        emb_raw = emb

        # Normalize per-sample to [0,1] for injection/encoding
        emb_norm = _minmax_per_sample(emb_raw)

        if getattr(model, "use_cnn_embed", False):
            x_seq = emb_norm.unsqueeze(0).repeat(T, 1, 1)   # [T, B, D]
        else:
            # Frozen ResNet path: stochastic spikes are fine
            x_seq = spikegen.rate(emb_norm, num_steps=T)    # [T, B, D]

        if return_embeddings:
            return x_seq, emb_raw.detach()
        return x_seq

    # No feature extractor: original path (flatten image -> encode_input)
    images = images.view(images.size(0), -1).to(device, non_blocking=True)
    x_seq = encode_input(images, T=T, mode=enc_mode, scale=enc_scale, bias=enc_bias).to(device)
    
    if return_embeddings:
        return x_seq, None
    return x_seq

# -----------------------------
# Conv edge pruning (channel-level) utilities
# edges: c -> c'  (kernel W[c', c, :, :])
# -----------------------------

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

class ChannelMaskedConv2d(nn.Module):
    """
    Wraps an nn.Conv2d and applies a channel-edge mask [C_out, C_in].
    Effective weight: W_eff = W * mask[:, :, None, None]
    Supports groups == 1 only.
    """
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("ChannelMaskedConv2d expects nn.Conv2d")
        if conv.groups != 1:
            raise NotImplementedError("Only groups==1 supported for now.")
        self.conv = conv
        self.register_buffer("mask", torch.ones(conv.out_channels, conv.in_channels, dtype=torch.float32))

    def forward(self, x):
        w_eff = self.conv.weight * self.mask[:, :, None, None]
        return nn.functional.conv2d(
            x, w_eff, self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups
        )

    @property
    def weight(self):
        return self.conv.weight


def iter_masked_convs(model: nn.Module) -> Iterable[Tuple[str, ChannelMaskedConv2d]]:
    for name, m in model.named_modules():
        if isinstance(m, ChannelMaskedConv2d):
            yield name, m


def wrap_conv2d_with_edge_masks_(model: nn.Module, include_names: Optional[List[str]] = None) -> nn.Module:
    """
    In-place replace eligible nn.Conv2d with ChannelMaskedConv2d.
    include_names: if provided, wrap only modules whose full name contains any substring in list.
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


@torch.no_grad()
def init_random_sparsity_(layer: ChannelMaskedConv2d, sparsity: float, generator: Optional[torch.Generator] = None):
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0,1)")
    total = layer.mask.numel()
    num_prune = int(round(sparsity * total))
    layer.mask.fill_(1.0)
    if num_prune <= 0:
        return
    perm = torch.randperm(total, generator=generator, device=layer.mask.device)
    layer.mask.view(-1)[perm[:num_prune]] = 0.0


@torch.no_grad()
def edge_importance_l1(layer: ChannelMaskedConv2d) -> torch.Tensor:
    # [C_out, C_in]
    return layer.weight.abs().sum(dim=(2, 3))


@torch.no_grad()
def prune_low_importance_active_(layer: ChannelMaskedConv2d, prune_frac: float) -> int:
    if prune_frac <= 0:
        return 0
    m = layer.mask
    active = m.bool()
    n_active = int(active.sum().item())
    if n_active == 0:
        return 0
    n_prune = int(prune_frac * n_active)
    if n_prune < 1:
        return 0

    imp = edge_importance_l1(layer)
    active_imp = imp[active].view(-1)
    k = min(n_prune, active_imp.numel())
    thr, _ = torch.kthvalue(active_imp, k)
    to_prune = (imp <= thr) & active

    pruned = int(to_prune.sum().item())
    m[to_prune] = 0.0
    return pruned


@torch.no_grad()
def rewire_random_inactive_(layer: ChannelMaskedConv2d, num_to_grow: int, generator: Optional[torch.Generator] = None) -> int:
    if num_to_grow <= 0:
        return 0
    m = layer.mask
    inactive = (~m.bool())
    idx = inactive.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return 0
    n = min(num_to_grow, idx.size(0))
    perm = torch.randperm(idx.size(0), generator=generator, device=m.device)[:n]
    grow = idx[perm]
    m[grow[:, 0], grow[:, 1]] = 1.0
    return int(n)


@torch.no_grad()
def dynamic_prune_and_rewire_(layer: ChannelMaskedConv2d, prune_frac: float, generator: Optional[torch.Generator] = None) -> Dict[str, int]:
    pruned = prune_low_importance_active_(layer, prune_frac)
    grown = rewire_random_inactive_(layer, pruned, generator=generator)
    return {"pruned": pruned, "grown": grown}


@torch.no_grad()
def pre_training_prune_model_(model: nn.Module, sparsity: float, seed: Optional[int] = None):
    for name, layer in iter_masked_convs(model):
        g = None
        if seed is not None:
            g = torch.Generator(device=layer.mask.device)
            g.manual_seed(abs(hash((seed, name, "pre"))) % (2**31))
        init_random_sparsity_(layer, sparsity, generator=g)


@torch.no_grad()
def dynamic_pruning_step_model_(model: nn.Module, prune_frac: float, seed: Optional[int] = None) -> Dict[str, Dict[str, int]]:
    out = {}
    for name, layer in iter_masked_convs(model):
        g = None
        if seed is not None:
            g = torch.Generator(device=layer.mask.device)
            g.manual_seed(abs(hash((seed, name, "dyn"))) % (2**31))
        out[name] = dynamic_prune_and_rewire_(layer, prune_frac, generator=g)
    return out


@torch.no_grad()
def post_training_prune_model_(model: nn.Module, prune_frac_of_active: float) -> Dict[str, int]:
    out = {}
    for name, layer in iter_masked_convs(model):
        out[name] = prune_low_importance_active_(layer, prune_frac_of_active)
    return out


def train_one_epoch(model, loader, optimizer, device, epoch_idx, use_dst, 
                   enforce_sparsity=False, lambda_coef=0.0, target_rate=0.09):
    """
    Train for one epoch. Simplified to always use CrossEntropy on spk_sum.
    """
    global global_step
    model.train()
    
    # Keep feature extractor in train mode if it's trainable CNN, eval if frozen ResNet
    if hasattr(model, 'feature_extractor') and model.feature_extractor is not None:
        if getattr(model, "use_resnet", False):
            model.feature_extractor.eval()
        else:
            model.feature_extractor.train()
    
    criterion_ce = nn.CrossEntropyLoss()
    
    total = 0
    correct = 0
    total_batches = len(loader)
    
    for batch_idx, (images, labels) in enumerate(loader):
        if IS_TTY:
            progress = (batch_idx + 1) / total_batches
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            percent = int(progress * 100)
            end_char = "\r" if (batch_idx + 1) < total_batches else "\n"
            print(
                f"[Epoch {epoch_idx}] [{bar}] {percent:3d}% ({batch_idx + 1}/{total_batches})",
                end=end_char,
                flush=True,
            )

        labels = labels.to(device, non_blocking=True)
        x_seq = _make_input_sequence(images, device, model)
        
        optimizer.zero_grad()
        
        # Forward pass - different return values based on needs
        if enforce_sparsity and lambda_coef > 0:
            spk_sum, hidden_spikes = model(x_seq, return_hidden_spikes=True)
        elif use_dst and isinstance(model, SPARSE_MODEL_TYPES):
            spk_sum, acts = model(x_seq, return_activations=True)
            update_hebb_buffer(x_seq, acts)
            hidden_spikes = None
        else:
            spk_sum = model(x_seq)
            hidden_spikes = None
        
        # Always use CrossEntropy on spike sums
        loss = criterion_ce(spk_sum, labels)
        preds = spk_sum.argmax(dim=1)
        
        # Add firing rate regularization if enabled
        if enforce_sparsity and lambda_coef > 0 and hidden_spikes is not None:
            fr_loss = firing_rate_loss(
                [hidden_spikes["layer1"], hidden_spikes["layer2"], hidden_spikes["layer3"]],
                target=target_rate,
                T_steps=T
            )
            loss = loss + lambda_coef * fr_loss
        
        loss.backward()
        optimizer.step()
        
        # Dynamic Sparse Training step
        if use_dst and isinstance(model, SPARSE_MODEL_TYPES):
            if global_step > 0 and global_step % UPDATE_INTERVAL == 0:
                dst_step(model)
        
        global_step += 1
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on test set. Simplified to always use spk_sum."""
    model.eval()
    
    if hasattr(model, 'feature_extractor') and model.feature_extractor is not None:
        model.feature_extractor.eval()
    
    total = 0
    correct = 0
    
    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        x_seq = _make_input_sequence(images, device, model)
        
        spk_sum = model(x_seq)
        preds = spk_sum.argmax(dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return correct / total


@torch.no_grad()
def compute_firing_rates(model, loader, device):
    """Compute average firing rates across hidden layers."""
    model.eval()
    if hasattr(model, 'feature_extractor') and model.feature_extractor is not None:
        model.feature_extractor.eval()
    
    total_samples = 0
    l1_sum = l2_sum = l3_sum = None
    
    for images, _ in loader:
        B = images.size(0)
        total_samples += B
        x_seq = _make_input_sequence(images, device, model)
        spk_sum, hidden_spikes = model(x_seq, return_hidden_spikes=True)
        
        b1 = hidden_spikes["layer1"].sum(dim=0)
        b2 = hidden_spikes["layer2"].sum(dim=0)
        b3 = hidden_spikes["layer3"].sum(dim=0)
        
        if l1_sum is None:
            l1_sum, l2_sum, l3_sum = b1, b2, b3
        else:
            l1_sum += b1
            l2_sum += b2
            l3_sum += b3
    
    denom = T * total_samples
    r1 = l1_sum / denom
    r2 = l2_sum / denom
    r3 = l3_sum / denom
    allr = torch.cat([r1, r2, r3])
    
    return {
        "layer1_mean": r1.mean().item(),
        "layer2_mean": r2.mean().item(),
        "layer3_mean": r3.mean().item(),
        "overall_hidden_mean": allr.mean().item(),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train SNN models on multiple datasets.")
    p.add_argument("--dataset", type=str, default="fashionmnist", 
                   choices=["fashionmnist", "cifar10", "cifar100"])
    p.add_argument("--model", type=str, default="dense", 
                   choices=["dense", "index", "random", "mixer"])
    p.add_argument("--epochs", type=int, default=num_epochs)
    p.add_argument("--p_inter", type=float, default=0.15)
    p.add_argument("--sparsity_mode", type=str, default="static", 
                   choices=["static", "dynamic"])
    p.add_argument("--cp", type=str, default="set", 
                   choices=["set", "random", "hebb"])
    p.add_argument("--cg", type=str, default="hebb", 
                   choices=["hebb", "random"])
    p.add_argument("--T", type=int, default=T)
    p.add_argument("--batch_size", type=int, default=batch_size)
    p.add_argument("--hidden_dim", type=int, default=hidden_dim)
    p.add_argument("--enc", type=str, default="current", 
                   choices=["current", "rate"])
    p.add_argument("--enc_scale", type=float, default=1.0)
    p.add_argument("--enc_bias", type=float, default=0.0)
    p.add_argument("--use_resnet", action="store_true")
    p.add_argument("--enforce_sparsity", action="store_true")
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--use_cnn_embed", action="store_true")
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--conv_channels", type=int, default=32)
    p.add_argument("--resnet_cardinality", type=int, default=4,
               help="Number of groups in grouped conv (ResNeXt cardinality)")
    p.add_argument("--resnet_width_per_group", type=int, default=4,
               help="Width per group for bottleneck: D = cardinality * width_per_group")


    return p.parse_args()



def main():
    args = parse_args()
    done_marker = get_done_marker_path(args)
    
    if os.path.exists(done_marker):
        print("[DONE] This experiment already finished. Skipping.")
        print(f"[INFO] To re-run, delete: {done_marker}")
        return
    
    device = select_device()
    
    # Update global variables
    global cp_mode, cg_mode, T, batch_size, hidden_dim
    cp_mode = args.cp
    cg_mode = args.cg
    T = args.T
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    
    global enc_mode, enc_scale, enc_bias
    enc_mode = args.enc
    enc_scale = float(args.enc_scale)
    enc_bias = float(args.enc_bias)
    
    global input_dim, num_classes
    
    normalize_images = bool(args.use_resnet and not args.use_cnn_embed)
    
    # Load data
    if args.dataset == "fashionmnist":
        input_dim = 28 * 28
        num_classes = 10
        train_loader, test_loader = get_fashion_loaders(batch_size)
    elif args.dataset == "cifar10":
        input_dim = 3 * 32 * 32
        num_classes = 10
        train_loader, test_loader = get_cifar10_loaders(batch_size, normalize=normalize_images)
    elif args.dataset == "cifar100":
        input_dim = 3 * 32 * 32
        num_classes = 100
        train_loader, test_loader = get_cifar100_loaders(batch_size, normalize=normalize_images)
    
    # Print configuration
    num_groups = 8 if args.model in ["index", "random", "mixer"] else "N/A"
    
    if args.use_cnn_embed:
        feature_str = f"CNN-1L (ch={args.conv_channels}, emb={args.embedding_dim})"
    elif args.use_resnet:
        feature_str = "ResNet-32 (frozen)"
    else:
        feature_str = "Direct"
    
    inject_str = "raw" if (args.use_resnet or args.use_cnn_embed) else enc_mode
    norm_str = "normalized" if normalize_images else "raw [0,1]"
    
    print(
        f"[CONFIG] dataset={args.dataset} | model={args.model} | feature_extract={feature_str} | "
        f"preprocessing={norm_str} | inject={inject_str} | "
        f"sparsity={args.sparsity_mode} | T={T} | epochs={args.epochs} | "
        f"batch={batch_size} | hidden_dim={hidden_dim} | groups={num_groups} | p_inter={args.p_inter}"
    )
    print("=" * 70)
    
    # Build model 
    model = build_model(args.model, args.p_inter, args.dataset, use_resnet=args.use_resnet, args=args).to(device)
    
    # Move feature extractor to device if it exists
    if getattr(model, "feature_extractor", None) is not None:
        model.feature_extractor = model.feature_extractor.to(device)
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    
    use_dst = (args.sparsity_mode == "dynamic")
    start_epoch = load_checkpoint(model, optimizer, args)
    
    if start_epoch > 1:
        model = model.to(device)
    
    last_completed_epoch = start_epoch - 1
    
    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if args.enforce_sparsity:
                lambda_coef, target_rate = get_progressive_params(epoch, args.epochs, args.warmup_epochs)
                if epoch == args.warmup_epochs + 1:
                    print(f"\n[Sparsity enforcement starting from epoch {epoch}]")
                if epoch > args.warmup_epochs and epoch % 10 == 0:
                    print(f"[Sparsity] lambda={lambda_coef:.6f}, target_rate={target_rate:.4f}")
            else:
                lambda_coef, target_rate = 0.0, 0.09
            
            train_acc = train_one_epoch(
                model, train_loader, optimizer, device, epoch, 
                use_dst, args.enforce_sparsity, lambda_coef, target_rate
            )
            test_acc = evaluate(model, test_loader, device)
            
            print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")
            
            last_completed_epoch = epoch
            save_checkpoint(epoch, model, optimizer, args, {'train_acc': train_acc, 'test_acc': test_acc})
    
    except KeyboardInterrupt:
        print("\n[Training interrupted by user - checkpoint saved]")
        if last_completed_epoch >= start_epoch:
            save_checkpoint(last_completed_epoch, model, optimizer, args)
        return
    
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        if last_completed_epoch >= start_epoch:
            save_checkpoint(last_completed_epoch, model, optimizer, args)
        raise
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL METRICS")
    print("=" * 70)
    
    rates = compute_firing_rates(model, test_loader, device)
    print("\nAverage firing rates:")
    print(f" L1: {rates['layer1_mean']:.6f}")
    print(f" L2: {rates['layer2_mean']:.6f}")
    print(f" L3: {rates['layer3_mean']:.6f}")
    print(f" Overall: {rates['overall_hidden_mean']:.6f}")
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    
    with open(done_marker, 'w') as f:
        f.write("Training completed successfully\n")
    print(f"[DONE marker created: {done_marker}]")


if __name__ == "__main__":
    main()