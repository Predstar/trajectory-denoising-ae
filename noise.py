"""Artificial noise injection for trajectory data."""

import numpy as np


def add_gaussian_noise(trajectories, sigma=0.1):
    """Add i.i.d. Gaussian noise to trajectories.

    Args:
        trajectories: np.ndarray of shape (N, T, 3) or (T, 3).
        sigma: standard deviation of the noise.

    Returns:
        Noisy trajectories, same shape as input.
    """
    noise = np.random.normal(0, sigma, trajectories.shape)
    return trajectories + noise
