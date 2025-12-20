import os
import sys
import argparse

# Allow imports from project root 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import FashionMNIST
from torchvision import transforms

# Import utilities and hyperparameters from train.py
from evaluation.train import (
    build_model,
    train_one_epoch,
    evaluate,
    select_device,
    batch_size,
    lr,
    T,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_fashion_train_dataset():
    """Return the Fashion-MNIST training dataset (no test split here)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = FashionMNIST(
        root=os.path.join(ROOT_DIR, "data"),
        train=True,
        download=True,
        transform=transform,
    )
    return dataset


def run_cross_validation(
    model_name: str,
    p_inter: float,
    epochs: int = 5,
    k_folds: int = 5,
):
    """
    Run k-fold cross-validation on the Fashion-MNIST training set.

    For each fold:
        - Build a fresh model
        - Train for 'epochs' epochs on 4/5 of the data
        - Evaluate on the remaining 1/5
        - Store final train and validation accuracies

    At the end, print mean ± std over folds.
    """
    set_seed(42)

    device = select_device()
    print(f"Cross-validation on device: {device}")

    # Load full training dataset 
    full_dataset = get_fashion_train_dataset()
    num_samples = len(full_dataset)
    indices = list(range(num_samples))

    # Fold size 
    fold_size = num_samples // k_folds

    fold_train_accs = []
    fold_val_accs = []

    base_msg = (
        f"\nRunning {k_folds}-fold CV | model={model_name}, "
        f"epochs={epochs}, batch_size={batch_size}"
    )

    if model_name in {"index", "random", "mixer"}:
        base_msg = (
            f"\nRunning {k_folds}-fold CV | model={model_name}, "
            f"p_inter={p_inter}, epochs={epochs}, batch_size={batch_size}"
        )

    print(base_msg)


    for fold in range(k_folds):
        print(f"Fold {fold + 1}/{k_folds}")

        # Determine validation indices for this fold
        val_start = fold * fold_size
        # Last fold takes all remaining samples
        val_end = (fold + 1) * fold_size if fold < k_folds - 1 else num_samples

        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        # Samplers for DataLoaders
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
        )

        val_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
        )

        # Build a fresh model for this fold
        model = build_model(model_name=model_name, p_inter=p_inter).to(device)

        # Optimizer for this model
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        last_train_acc = None
        last_val_acc = None

        # Train for the specified number of epochs
        for epoch in range(1, epochs + 1):
            train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch_idx=epoch)
            val_acc = evaluate(model, val_loader, device)

            last_train_acc = train_acc
            last_val_acc = val_acc

            print(
                f"Fold {fold + 1} | "
                f"Epoch {epoch:02d}/{epochs} "
                f"| train_acc={train_acc * 100:.2f}% "
                f"| val_acc={val_acc * 100:.2f}%"
            )

        # Store final accuracies for this fold (after the last epoch)
        fold_train_accs.append(last_train_acc)
        fold_val_accs.append(last_val_acc)

    # Convert to numpy arrays for convenience
    fold_train_accs = np.array(fold_train_accs)
    fold_val_accs = np.array(fold_val_accs)

    # Compute mean and standard deviation across folds
    train_mean = fold_train_accs.mean() * 100.0
    train_std = fold_train_accs.std() * 100.0
    val_mean = fold_val_accs.mean() * 100.0
    val_std = fold_val_accs.std() * 100.0

    print("Cross-validation summary")
    print(f"Train accuracy over folds: {train_mean:.2f}% ± {train_std:.2f}%")
    print(f"Val   accuracy over folds: {val_mean:.2f}% ± {val_std:.2f}%")

    return {
        "fold_train_accs": fold_train_accs,
        "fold_val_accs": fold_val_accs,
        "train_mean": train_mean,
        "train_std": train_std,
        "val_mean": val_mean,
        "val_std": val_std,
    }


def parse_args():
    """Parse command-line arguments for cross-validation."""
    parser = argparse.ArgumentParser(
        description="5-fold cross-validation for SNN models on Fashion-MNIST."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dense",
        choices=["dense", "index", "random", "mixer"],
        help="Model type to evaluate.",
    )

    parser.add_argument(
        "--p_inter",
        type=float,
        default=0.15,
        help="Inter-group connection probability p' for sparse models "
             "(Index, Random, Mixer). Ignored for the dense model.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs per fold.",
    )

    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_cross_validation(
        model_name=args.model,
        p_inter=args.p_inter,
        epochs=args.epochs,
        k_folds=args.k_folds,
    )


if __name__ == "__main__":
    main()
