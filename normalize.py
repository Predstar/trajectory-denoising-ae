"""Per-trajectory scale normalization (thesis Section 4.3 ablation).

Each trajectory is divided by its own scale factor before being fed to the
encoder, and the decoder's output is multiplied back by that same factor
before computing MSE in original units. The scale is computed from the
noisy trajectory (what the encoder actually sees at inference time), not
the clean one.
"""

import numpy as np
import torch


def compute_scale(noisy_flat, T):
    """Per-trajectory scale factor d_max = max_t ||x_t||_2.

    Args:
        noisy_flat: torch.Tensor (N, T*3), flattened noisy trajectories.
        T: timesteps per trajectory.

    Returns:
        torch.Tensor (N, 1) of per-trajectory scale factors, safe to divide by.
    """
    noisy = noisy_flat.reshape(-1, T, 3)
    norms = noisy.norm(dim=2)                 # (N, T)
    d_max = norms.max(dim=1, keepdim=True).values  # (N, 1)
    return d_max.clamp(min=1e-8)               # avoid div-by-zero


def normalize(flat, scale):
    """Divide each trajectory by its own scale factor."""
    return flat / scale


def denormalize(flat, scale):
    """Multiply each trajectory back by its own scale factor."""
    return flat * scale
