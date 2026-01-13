# G2G_Sparse_SNN

Spiking Neural Network implementations exploring different connectivity patterns inspired by biological neural circuits:

- **Dense**: Fully-connected baseline for comparison
- **Index**: G2GNet with index-based grouping (preserves spatial locality)
- **Random**: G2GNet with random grouping (disrupts spatial structure)
- **Mixer**: G2GNet with mixer-based grouping (alternates between spatial and feature mixing)

**G2GNet** is our proposed architecture that uses sparse, modular connectivity inspired by ensemble-to-ensemble communication observed in mouse visual cortex. The three grouping strategies (Index, Random, Mixer) represent different ways to organize neurons within each layer.

You can train these models normally, enable **Dynamic Sparse Training (DST)** to update sparse connectivity during training, or run **k-fold cross-validation**.

---

## Installation

I'd recommend setting up a virtual environment first:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
```

### Core dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision snntorch numpy
```

### GPU support

**NVIDIA (CUDA)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**AMD GPU (Windows, DirectML)**
```bash
pip install torch-directml
```

The code will automatically detect and use: DirectML → CUDA → CPU (in that priority order).

---

## Training

Run the main training script:
```bash
python train.py --dataset DATASET --model MODEL [OPTIONS]
```

### Core Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--dataset` | Dataset to use | `fashionmnist` | `fashionmnist`, `cifar10`, `cifar100` |
| `--model` | Model architecture | `dense` | `dense`, `index`, `random`, `mixer` |
| `--epochs` | Training epochs | `20` | any int |
| `--p_inter` | Inter-group connection probability (sparse models only) | `0.15` | 0.0-1.0 |

### Sparse Training Options

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--sparsity_mode` | Sparsity mode | `static` | `static`, `dynamic` |
| `--cp` | Pruning rule for DST | `set` | `set`, `random`, `hebb` |
| `--cg` | Growth rule for DST | `hebb` | `hebb`, `random` |

**Pruning rules (--cp):**
- `set`: Magnitude-based pruning (SET)
- `random`: Random pruning
- `hebb`: Hebbian correlation-based pruning

**Growth rules (--cg):**
- `hebb`: Hebbian correlation-based growth (recommended)
- `random`: Random growth

### Network Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--T` | Simulation timesteps | `50` |
| `--batch_size` | Training batch size | `256` |
| `--hidden_dim` | Hidden layer dimension | `2048` |

### Encoding Options

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--enc` | Spike encoding method | `current` | `current`, `rate` |
| `--enc_scale` | Encoding scale factor | `1.0` | any float |
| `--enc_bias` | Encoding bias | `0.0` | any float |

### Advanced Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_resnet` | Use frozen ResNet-32 feature extractor (CIFAR only) | `False` |
| `--enforce_sparsity` | Enable firing rate regularization | `False` |
| `--warmup_epochs` | Warmup epochs before sparsity enforcement | `10` |

---

## Example Commands

### Standard Training (Static Sparsity)

**Fashion-MNIST**
```bash
# Dense baseline
python train.py --dataset fashionmnist --model dense --epochs 20

# Index-based sparse
python train.py --dataset fashionmnist --model index --p_inter 0.15

# Random sparse
python train.py --dataset fashionmnist --model random --p_inter 0.20 --epochs 30

# Mixer sparse
python train.py --dataset fashionmnist --model mixer --p_inter 0.15
```

**CIFAR-10**
```bash
# Dense baseline
python train.py --dataset cifar10 --model dense --epochs 30

# Mixer with ResNet feature extraction
python train.py --dataset cifar10 --model mixer --use_resnet --epochs 30

# Index with ResNet
python train.py --dataset cifar10 --model index --use_resnet --p_inter 0.20
```

**CIFAR-100**
```bash
# Mixer with ResNet
python train.py --dataset cifar100 --model mixer --use_resnet --p_inter 0.20 --epochs 40

# Random with ResNet
python train.py --dataset cifar100 --model random --use_resnet --p_inter 0.15
```

### Dynamic Sparse Training (DST)

**All sparse models (Index, Random, Mixer) support DST:**

**Hebbian pruning + Hebbian growth (recommended):**
```bash
python train.py --dataset fashionmnist --model index \
    --sparsity_mode dynamic --cp hebb --cg hebb --epochs 20

python train.py --dataset fashionmnist --model random \
    --sparsity_mode dynamic --cp hebb --cg hebb --epochs 20

python train.py --dataset fashionmnist --model mixer \
    --sparsity_mode dynamic --cp hebb --cg hebb --epochs 20
```

**Magnitude pruning (SET) + Hebbian growth:**
```bash
python train.py --dataset fashionmnist --model mixer \
    --sparsity_mode dynamic --cp set --cg hebb --epochs 20
```

**Magnitude pruning + Random growth:**
```bash
python train.py --dataset fashionmnist --model index \
    --sparsity_mode dynamic --cp set --cg random --epochs 20
```

**DST on CIFAR-10 with ResNet:**
```bash
python train.py --dataset cifar10 --model mixer --use_resnet \
    --sparsity_mode dynamic --cp hebb --cg hebb --epochs 30

python train.py --dataset cifar10 --model index --use_resnet \
    --sparsity_mode dynamic --cp set --cg hebb --epochs 30
```

### With Firing Rate Regularization
```bash
python train.py --dataset fashionmnist --model mixer \
    --enforce_sparsity --warmup_epochs 10 --epochs 30
```

---

## Cross-Validation
```bash
python crossval.py --dataset DATASET --model MODEL [OPTIONS]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset to use | `fashionmnist` |
| `--model` | Model type | `dense` |
| `--p_inter` | Inter-group connectivity (sparse models) | `0.15` |
| `--epochs` | Epochs per fold | `5` |
| `--k_folds` | Number of folds | `5` |

### Example
```bash
python crossval.py --dataset fashionmnist --model mixer --k_folds 5 --epochs 10
```

---

## Checkpointing & Resuming

The training script automatically:
- Saves checkpoints after each epoch to `checkpoints/`
- Resumes from the last checkpoint if interrupted
- Creates a `.DONE` marker when training completes
- Skips already-completed experiments

To re-run a completed experiment, delete the `.DONE` marker file.

**Checkpoint naming convention:**
- Static sparsity: `{dataset}_{model}_p{p_inter}_T{T}_{enc}.pth`
- Dynamic sparsity: `{dataset}_{model}_p{p_inter}_T{T}_{enc}_cp{cp}_cg{cg}.pth`
- With ResNet: `resnet_{dataset}_{model}_p{p_inter}_T{T}_{enc}.pth`

---

## Reproducibility

- Fixed random seed: `seed=42` (Python, NumPy, PyTorch)
- Deterministic CUDA operations enabled
- Reproducible cuDNN backend

---

## Datasets

### Fashion-MNIST
- **Resolution**: 28×28 grayscale
- **Classes**: 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Train/Test**: 60,000 / 10,000

### CIFAR-10
- **Resolution**: 32×32 RGB
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Train/Test**: 50,000 / 10,000

### CIFAR-100
- **Resolution**: 32×32 RGB
- **Classes**: 100 (grouped into 20 superclasses)
- **Train/Test**: 50,000 / 10,000

---

## Architecture Details

### Dense Model
- **Structure**: Input → FC(2048) → FC(2048) → FC(num_classes)
- **Connectivity**: Fully connected
- **Parameters**: ~4M (Fashion-MNIST)

### Sparse Models (Index/Random/Mixer)
- **Structure**: Input → Sparse(2048) → Sparse(2048) → Sparse(num_classes)
- **Groups**: 8 groups per layer
- **Intra-group**: Fully connected within each group (p_intra = 1.0)
- **Inter-group**: Sparse connections controlled by `p_inter`
- **Parameters**: ~1-2M depending on `p_inter`

### Feature Extraction (CIFAR)
- **Direct**: Flatten raw pixels (3×32×32 = 3072)
- **ResNet-32**: Frozen pretrained feature extractor → truncated at layer2 → global average pooling (4×4) → output: 512-dim

---

## About G2GNet and DST

### G2GNet Architecture
G2GNet is inspired by ensemble-to-ensemble connectivity observed in mouse visual cortex. Key principles:

1. **Sparse, modular connectivity** reduces parameter count while maintaining performance
2. **Group-based organization** mimics cortical ensemble structure (8 groups per layer)
3. **Flexible grouping strategies**:
   - **Index**: Preserves spatial locality (neurons grouped by index)
   - **Random**: Disrupts spatial structure (random assignment)
   - **Mixer**: Alternates between spatial and feature mixing

### Dynamic Sparse Training (DST)

DST dynamically reallocates sparse connections during training to improve performance:

- **Update frequency**: Every 1,000 training steps
- **Prune fraction**: 10% of existing connections removed per update
- **Growth fraction**: Equal number of new connections added

**Pruning strategies (C_P):**
- `set`: Magnitude-based (removes smallest weights)
- `random`: Random removal
- `hebb`: Based on Hebbian correlation (removes weakly correlated connections)

**Growth strategies (C_G):**
- `hebb`: Based on neuron correlation - adds connections where pre/post activity is correlated (recommended)
- `random`: Random growth

**Implementation notes:**
- DST applies only to sparse layers (fc1, fc2, fc3) in sparse models
- Connections are reallocated while maintaining overall sparsity level
- Hebbian correlation is computed from recent spike activity buffers

### DST Compatibility

| Model | Static Sparse | Dynamic Sparse (DST) |
|-------|---------------|----------------------|
| Dense | N/A | ❌ No (fully connected) |
| Index | ✔ Yes | ✔ **Full support** |
| Random | ✔ Yes | ✔ **Full support** |
| Mixer | ✔ Yes | ✔ **Full support** |
