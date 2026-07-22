"""Synthetic 3D trajectory generation (spiral / circular motion)."""

import numpy as np


def generate_spiral_trajectory(T=200, t_max=4 * np.pi, r0=1.0, k=0.1, vz=0.3):
    """Generate a single 3D spiral trajectory.

    Args:
        T: number of timesteps.
        t_max: end of the time range (start is always 0).
        r0: starting radius.
        k: radius growth rate (positive=expanding, negative=contracting).
        vz: vertical (z) speed.

    Returns:
        np.ndarray of shape (T, 3) -- x, y, z coordinates over time.
    """
    t = np.linspace(0, t_max, T)
    r = r0 + k * t
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = vz * t
    return np.stack([x, y, z], axis=1)


def generate_dataset(n_trajectories=100, T=200, param_ranges=None):
    """Generate a batch of synthetic trajectories with randomized parameters.

    Args:
        n_trajectories: number of trajectories to generate.
        T: timesteps per trajectory.
        param_ranges: optional dict specifying (min, max) ranges for
            r0, k, vz, t_max to randomize across the dataset.

    Returns:
        np.ndarray of shape (n_trajectories, T, 3).
    """
    if param_ranges is None:
        param_ranges = {
            "r0": (0.5, 1.5),
            "k": (-0.2, 0.2),
            "vz": (-0.5, 0.5),
            "t_max": (2 * np.pi, 6 * np.pi),
        }

    trajectories = []
    for _ in range(n_trajectories):
        r0 = np.random.uniform(*param_ranges["r0"])
        k = np.random.uniform(*param_ranges["k"])
        vz = np.random.uniform(*param_ranges["vz"])
        t_max = np.random.uniform(*param_ranges["t_max"])
        trajectories.append(generate_spiral_trajectory(T=T, t_max=t_max, r0=r0, k=k, vz=vz))

    return np.stack(trajectories, axis=0)
