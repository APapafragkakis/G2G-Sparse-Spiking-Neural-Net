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
from data.fashionmnist import get_fashion_loaders
from data.cifar10_100 import get_cifar10_loaders, get_cifar100_loaders
from utils.encoding import encode_input
import warnings

warnings.filterwarnings("ignore", message=".*aten::lerp.Scalar_out.*")

# Sparse layer and model types for DST
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


# Default training parameters (can be overridden via CLI)
batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024
hidden_dim_dense = 447
num_classes = 10
num_epochs = 20
lr = 1e-3

# Input encoding configuration (set from CLI)
enc_mode = "current"
enc_scale = 1.0
enc_bias = 0.0

# Dynamic Sparse Training (DST) configuration
global_step = 0
UPDATE_INTERVAL = 1000
cp_mode = "set"
cg_mode = "hebb"

# Buffers for Hebbian pre/post activity
hebb_buffer = {"fc1": None, "fc2": None, "fc3": None}

# Check if output is redirected (for .bat script compatibility)
IS_TTY = sys.stdout.isatty()


def get_checkpoint_path(args):
    """Generate checkpoint filename based on configuration"""
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Format p_inter to avoid messy decimals
    p_str = f"{args.p_inter:.2f}".replace(".", "_")
    
    # Include cp/cg only if using dynamic sparsity
    if args.sparsity_mode == "dynamic":
        filename = f"{args.dataset}_{args.model}_p{p_str}_T{args.T}_{args.enc}_cp{args.cp}_cg{args.cg}.pth"
    else:
        filename = f"{args.dataset}_{args.model}_p{p_str}_T{args.T}_{args.enc}.pth"
    
    return os.path.join(checkpoint_dir, filename)


def get_done_marker_path(args):
    """Generate DONE marker filename to track completed runs"""
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    p_str = f"{args.p_inter:.2f}".replace(".", "_")
    
    # Include cp/cg only if using dynamic sparsity
    if args.sparsity_mode == "dynamic":
        filename = f"{args.dataset}_{args.model}_p{p_str}_T{args.T}_{args.enc}_cp{args.cp}_cg{args.cg}.DONE"
    else:
        filename = f"{args.dataset}_{args.model}_p{p_str}_T{args.T}_{args.enc}.DONE"
    
    return os.path.join(checkpoint_dir, filename)


def save_checkpoint(epoch, model, optimizer, args, metrics=None):
    """Save training checkpoint"""
    checkpoint_path = get_checkpoint_path(args)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'global_step': global_step,
    }
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, checkpoint_path)
    print(f"[Checkpoint saved: epoch {epoch}]")


def load_checkpoint(model, optimizer, args):
    """Load training checkpoint if it exists"""
    checkpoint_path = get_checkpoint_path(args)
    
    if not os.path.exists(checkpoint_path):
        return 1  # Start from epoch 1 (not 0!)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        global global_step
        global_step = checkpoint.get('global_step', 0)
        
        start_epoch = checkpoint['epoch'] + 1  # Continue from next epoch
        print(f"[Checkpoint loaded: resuming from epoch {start_epoch}]")
        
        return start_epoch
    except Exception as e:
        print(f"[Warning: Failed to load checkpoint: {e}]")
        return 1


def build_model(model_name: str, p_inter: float):
    if model_name == "dense":
        return DenseSNN(input_dim, hidden_dim_dense, num_classes)
    elif model_name == "index":
        return IndexSNN(input_dim, hidden_dim, num_classes, 8, 1.0, p_inter)
    elif model_name == "random":
        return RandomSNN(input_dim, hidden_dim, num_classes, 8, 1.0, p_inter)
    elif model_name == "mixer":
        return MixerSNN(input_dim, hidden_dim, num_classes, 8, 1.0, p_inter)
    else:
        raise ValueError(f"Unknown model: {model_name}")


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
    if cp_mode_local == "set":
        active_weights = w_cpu[active].abs().view(-1)
        n_eff = min(total_to_prune, active_weights.numel())
        if n_eff < 1:
            return
        thresh, _ = torch.kthvalue(active_weights, n_eff)
        prune_mask = active & (w_cpu.abs() <= thresh)
    elif cp_mode_local == "random":
        active_idx = active.nonzero(as_tuple=False)
        n_eff = min(total_to_prune, active_idx.size(0))
        if n_eff < 1:
            return
        perm = torch.randperm(active_idx.size(0))[:n_eff]
        chosen = active_idx[perm]
        prune_mask = torch.zeros_like(mask_cpu, dtype=torch.bool)
        prune_mask[chosen[:, 0], chosen[:, 1]] = True
    else:  # Hebbian pruning
        if ch_cpu is None:
            return
        active_scores = ch_cpu[active].view(-1)
        n_eff = min(total_to_prune, active_scores.numel())
        if n_eff < 1:
            return
        thresh, _ = torch.kthvalue(active_scores, n_eff)
        prune_mask = active & (ch_cpu <= thresh)
    mask_cpu[prune_mask] = 0
    num_pruned = prune_mask.sum().item()
    if ch_cpu is None:
        ch_cpu = torch.zeros_like(mask_cpu, dtype=torch.float32)
    mask_cpu, _ = _grow_connections(mask_cpu, ch_cpu, num_pruned, cg_mode_local)
    mask.copy_(mask_cpu.to(device))
    weight[~mask.bool()] = 0.0


def dst_step(model, prune_frac=0.025):
    for name, module in model.named_modules():
        if isinstance(module, SPARSE_LAYER_TYPES):
            short = name.split(".")[-1]
            if short in hebb_buffer:
                dst_update_layer_cp_cg_single(module, short, prune_frac, cp_mode, cg_mode)
    print("[DST] step executed")


def _make_input_sequence(images, device):
    return encode_input(
        images,
        T,
        mode=enc_mode,
        scale=enc_scale,
        bias=enc_bias,
    ).to(device)


def train_one_epoch(model, loader, optimizer, device, epoch_idx, use_dst):
    global global_step
    model.train()
    total = 0
    correct = 0
    total_batches = len(loader)
    
    for batch_idx, (images, labels) in enumerate(loader):
        # Progress bar - only show if output is to terminal
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
        x_seq = _make_input_sequence(images, device)
        optimizer.zero_grad()
        if use_dst and isinstance(model, SPARSE_MODEL_TYPES):
            spk_counts, acts = model(x_seq, return_activations=True)
            update_hebb_buffer(x_seq, acts)
        else:
            spk_counts = model(x_seq)
        loss = nn.CrossEntropyLoss()(spk_counts, labels)
        loss.backward()
        optimizer.step()
        if use_dst and isinstance(model, SPARSE_MODEL_TYPES):
            if global_step > 0 and global_step % UPDATE_INTERVAL == 0:
                dst_step(model)
        global_step += 1
        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        x_seq = _make_input_sequence(images, device)
        spk_counts = model(x_seq)
        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def compute_firing_rates(model, loader, device):
    model.eval()
    total_samples = 0
    l1_sum = l2_sum = l3_sum = None
    for images, _ in loader:
        B = images.size(0)
        total_samples += B
        x_seq = _make_input_sequence(images, device)
        _, hidden_spikes = model(x_seq, return_hidden_spikes=True)
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
    p.add_argument("--dataset", type=str, default="fashionmnist", choices=["fashionmnist", "cifar10", "cifar100"])
    p.add_argument("--model", type=str, default="dense", choices=["dense", "index", "random", "mixer"])
    p.add_argument("--epochs", type=int, default=num_epochs)
    p.add_argument("--p_inter", type=float, default=0.15)
    p.add_argument("--sparsity_mode", type=str, default="static", choices=["static", "dynamic"])
    p.add_argument("--cp", type=str, default="set", choices=["set", "random", "hebb"])
    p.add_argument("--cg", type=str, default="hebb", choices=["hebb", "random"])
    p.add_argument("--T", type=int, default=T)
    p.add_argument("--batch_size", type=int, default=batch_size)
    p.add_argument("--enc", type=str, default="current", choices=["current", "rate"])
    p.add_argument("--enc_scale", type=float, default=1.0)
    p.add_argument("--enc_bias", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    
    # Check if already completed
    done_marker = get_done_marker_path(args)
    if os.path.exists(done_marker):
        print("[DONE] This experiment already finished. Skipping.")
        print(f"[INFO] To re-run, delete: {done_marker}")
        return
    
    device = select_device()
    
    global cp_mode, cg_mode, T, batch_size
    cp_mode = args.cp
    cg_mode = args.cg
    T = args.T
    batch_size = args.batch_size
    
    global enc_mode, enc_scale, enc_bias
    enc_mode = args.enc
    enc_scale = float(args.enc_scale)
    enc_bias = float(args.enc_bias)
    
    global input_dim, num_classes
    if args.dataset == "fashionmnist":
        input_dim = 28 * 28
        num_classes = 10
        train_loader, test_loader = get_fashion_loaders(batch_size)
    elif args.dataset == "cifar10":
        input_dim = 3 * 32 * 32
        num_classes = 10
        train_loader, test_loader = get_cifar10_loaders(batch_size)
    elif args.dataset == "cifar100":
        input_dim = 3 * 32 * 32
        num_classes = 100
        train_loader, test_loader = get_cifar100_loaders(batch_size)
    
    num_groups = 8 if args.model in ["index", "random", "mixer"] else "N/A"
    print(
        f"[CONFIG] dataset={args.dataset} | model={args.model} | "
        f"sparsity={args.sparsity_mode} | T={T} | epochs={args.epochs} | "
        f"batch={batch_size} | groups={num_groups} | p_inter={args.p_inter} | "
        f"enc={enc_mode} | enc_scale={enc_scale} | enc_bias={enc_bias}"
    )
    print("=" * 70)
    
    model = build_model(args.model, args.p_inter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    use_dst = (args.sparsity_mode == "dynamic")
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, optimizer, args)
    if start_epoch > 1:
        model = model.to(device)  # Ensure model is on correct device after loading
    
    # Track last completed epoch for safe interrupt handling
    last_completed_epoch = start_epoch - 1
    
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, use_dst)
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")
            
            # Update last completed epoch
            last_completed_epoch = epoch
            
            # Save checkpoint after each successful epoch
            save_checkpoint(epoch, model, optimizer, args, {
                'train_acc': train_acc,
                'test_acc': test_acc
            })
            
    except KeyboardInterrupt:
        print("\n[Training interrupted by user - checkpoint saved]")
        # Save the last successfully completed epoch
        if last_completed_epoch >= start_epoch:
            save_checkpoint(last_completed_epoch, model, optimizer, args)
        return
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        # Save the last successfully completed epoch
        if last_completed_epoch >= start_epoch:
            save_checkpoint(last_completed_epoch, model, optimizer, args)
        raise
    
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
    
    # Mark as done (checkpoint remains for reproducibility)
    with open(done_marker, 'w') as f:
        f.write("Training completed successfully\n")
    print(f"[DONE marker created: {done_marker}]")


if __name__ == "__main__":
    main()