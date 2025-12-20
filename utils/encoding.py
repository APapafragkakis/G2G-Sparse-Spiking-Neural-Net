import torch

def encode_input(images, T, mode="current", scale=1.0, bias=0.0):
    """
    mode="current": analog/current injection (no sampling)
    mode="rate": Bernoulli rate coding (sampling)
    """
    B = images.size(0)
    x = images.view(B, -1).float()

    x = scale * x + bias

    if mode == "current":
        # keep analog values as currents (recommended to keep within a reasonable range)
        return x.unsqueeze(0).repeat(T, 1, 1)

    if mode == "rate":
        # Bernoulli expects probabilities in [0, 1]
        p = x.clamp(0.0, 1.0)
        return torch.bernoulli(p.unsqueeze(0).repeat(T, 1, 1))

    raise ValueError(f"Unknown encoding mode: {mode}")
