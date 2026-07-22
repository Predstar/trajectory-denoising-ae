"""Plotting utilities: trajectories, noise comparison, training curves."""

import numpy as np
import matplotlib.pyplot as plt

# Fixed categorical slots (blue, orange, aqua) -- see dataviz skill palette.
COLOR_CLEAN = "#2a78d6"
COLOR_NOISY = "#eb6834"
COLOR_RECON = "#1baf7a"


def plot_trajectory_3d(trajectory, ax=None, label=None, **kwargs):
    """Plot a single (T, 3) trajectory in 3D.

    Args:
        trajectory: np.ndarray (T, 3).
        ax: optional existing Axes3D to draw on.
        label: legend label.

    Returns:
        the Axes3D used.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=label, **kwargs)
    return ax


def plot_clean_noisy_reconstructed(clean, noisy, reconstructed):
    """Overlay clean vs noisy vs AE-reconstructed trajectory for comparison."""
    ax = plot_trajectory_3d(clean, label='clean', lw=2, color=COLOR_CLEAN)
    plot_trajectory_3d(noisy, ax=ax, label='noisy', lw=1, alpha=0.6, color=COLOR_NOISY)
    plot_trajectory_3d(reconstructed, ax=ax, label='reconstructed', lw=2, linestyle='--', color=COLOR_RECON)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    return ax


def plot_per_axis_timeseries(clean, noisy, reconstructed):
    """Plot x(t), y(t), z(t) as three stacked subplots for one trajectory.

    Args:
        clean, noisy, reconstructed: np.ndarray (T, 3).

    Returns:
        the array of Axes used.
    """
    t = np.arange(clean.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = ['x', 'y', 'z']
    for i, (ax, axis_label) in enumerate(zip(axes, labels)):
        ax.plot(t, clean[:, i], label='clean', lw=2, color=COLOR_CLEAN)
        ax.plot(t, noisy[:, i], label='noisy', lw=1, alpha=0.6, color=COLOR_NOISY)
        ax.plot(t, reconstructed[:, i], label='reconstructed', lw=2, linestyle='--', color=COLOR_RECON)
        ax.set_ylabel(axis_label)
    axes[0].legend(loc='upper right')
    axes[-1].set_xlabel('timestep')
    fig.suptitle('Per-axis trajectory comparison')
    fig.tight_layout()
    return axes


def plot_residuals(clean, reconstructed):
    """Plot (reconstructed - clean) per axis over time, to reveal systematic bias.

    Args:
        clean, reconstructed: np.ndarray (T, 3).

    Returns:
        the array of Axes used.
    """
    residual = reconstructed - clean
    t = np.arange(clean.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = ['x', 'y', 'z']
    colors = [COLOR_CLEAN, COLOR_NOISY, COLOR_RECON]
    for i, (ax, axis_label, c) in enumerate(zip(axes, labels, colors)):
        ax.axhline(0, color="#898781", lw=1)
        ax.plot(t, residual[:, i], lw=1.5, color=c)
        ax.set_ylabel(f'{axis_label} residual')
    axes[-1].set_xlabel('timestep')
    fig.suptitle('Reconstruction residual (reconstructed - clean)')
    fig.tight_layout()
    return axes


def plot_latent_space_pca(embeddings, color_by=None, color_label=None):
    """Reduce latent embeddings to 2D via PCA and scatter, optionally colored
    by a trajectory parameter (e.g. radius or rise rate).

    Args:
        embeddings: np.ndarray (N, embedding_size).
        color_by: optional np.ndarray (N,) of values to color points by.
        color_label: label for the colorbar, if color_by is given.

    Returns:
        the Axes used.
    """
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    # SVD-based PCA -- avoids a sklearn dependency for a 2-component projection.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(7, 6))
    if color_by is not None:
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=color_by, cmap='Blues', s=10, alpha=0.8)
        cbar = fig.colorbar(sc, ax=ax)
        if color_label:
            cbar.set_label(color_label)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], color=COLOR_CLEAN, s=10, alpha=0.8)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Latent space (PCA projection)')
    return ax


def plot_mse_histogram(clean, reconstructed):
    """Histogram of per-trajectory MSE across a test set.

    Args:
        clean, reconstructed: np.ndarray (N, T, 3) or (N, T*3).

    Returns:
        the Axes used.
    """
    clean = clean.reshape(clean.shape[0], -1)
    reconstructed = reconstructed.reshape(reconstructed.shape[0], -1)
    per_traj_mse = ((reconstructed - clean) ** 2).mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(per_traj_mse, bins=40, color=COLOR_CLEAN, edgecolor="#fcfcfb")
    ax.axvline(per_traj_mse.mean(), color=COLOR_NOISY, linestyle='--', lw=2,
               label=f'mean = {per_traj_mse.mean():.4f}')
    ax.set_xlabel('per-trajectory MSE')
    ax.set_ylabel('count')
    ax.set_title('Distribution of test-set reconstruction MSE')
    ax.legend()
    return ax


def plot_snr_improvement(clean, noisy, reconstructed, n_samples=30, seed=0):
    """Bar chart comparing per-sample SNR before (noisy) vs after (reconstructed)
    denoising, for a random subset of trajectories.

    SNR (dB) = 10 * log10(signal_power / noise_power), where signal_power is
    the clean trajectory's power and noise_power is the MSE against clean.

    Args:
        clean, noisy, reconstructed: np.ndarray (N, T, 3) or (N, T*3).

    Returns:
        the Axes used.
    """
    clean = clean.reshape(clean.shape[0], -1)
    noisy = noisy.reshape(noisy.shape[0], -1)
    reconstructed = reconstructed.reshape(reconstructed.shape[0], -1)

    n = clean.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    idx.sort()

    signal_power = (clean[idx] ** 2).mean(axis=1)
    noise_power_before = ((noisy[idx] - clean[idx]) ** 2).mean(axis=1)
    noise_power_after = ((reconstructed[idx] - clean[idx]) ** 2).mean(axis=1)

    snr_before = 10 * np.log10(signal_power / noise_power_before)
    snr_after = 10 * np.log10(signal_power / noise_power_after)

    x = np.arange(len(idx))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, snr_before, width, label='before (noisy)', color=COLOR_NOISY)
    ax.bar(x + width / 2, snr_after, width, label='after (reconstructed)', color=COLOR_RECON)
    ax.set_xlabel('sample')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR improvement from denoising')
    ax.legend()
    return ax


def plot_encoder_weight_heatmap(model):
    """Visualize the first encoder layer's weights as a heatmap.

    Rows are hidden units, columns are flattened (timestep * 3) input
    features; reshaping the column axis into (T, 3) blocks shows which
    timesteps/axes each hidden unit attends to most.

    Args:
        model: TrajectoryAE instance.

    Returns:
        the Axes used.
    """
    first_linear = model.encoder[0]
    weights = first_linear.weight.detach().cpu().numpy()  # (hidden, T*3)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = np.abs(weights).max()
    im = ax.imshow(weights, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('input feature (timestep * 3 + axis)')
    ax.set_ylabel('hidden unit')
    ax.set_title('Encoder first-layer weights')
    fig.colorbar(im, ax=ax, label='weight')
    return ax


def plot_trajectory_overlay(trajectories, n_samples=200, seed=None):
    """Overlay many trajectories in one static 3D plot to show dataset spread.

    Args:
        trajectories: np.ndarray (N, T, 3).
        n_samples: max number of trajectories to draw (randomly chosen if
            trajectories has more than this many).
        seed: optional random seed for reproducible sample selection.

    Returns:
        the Axes3D used.
    """
    n = trajectories.shape[0]
    if n > n_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n_samples, replace=False)
    else:
        idx = np.arange(n)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection='3d')
    for i in idx:
        ax.plot(trajectories[i, :, 0], trajectories[i, :, 1], trajectories[i, :, 2],
                lw=0.7, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'{len(idx)} of {n} trajectories')
    return ax


def plot_training_curve(train_losses, test_losses=None):
    """Plot training loss vs epoch, optionally overlaid with test loss."""
    _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label='Train loss')
    if test_losses is not None:
        ax.plot(test_losses, label='Test loss', linestyle='--')
        ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('Training Progress')
    return ax


def plot_noise_sweep(sigmas, test_mses):
    """Plot test MSE vs noise level (sigma), characterizing model robustness.

    Args:
        sigmas: sequence of NOISE_SIGMA values used.
        test_mses: sequence of final test MSE for each sigma (same order).

    Returns:
        the Axes used.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sigmas, test_mses, marker='o', color=COLOR_CLEAN, lw=2)
    ax.set_xlabel('noise sigma')
    ax.set_ylabel('final test MSE')
    ax.set_title('Reconstruction error vs noise level')
    return ax


def plot_bottleneck_ablation(embedding_sizes, test_mses):
    """Plot test MSE vs embedding (bottleneck) size, showing the
    compression-quality tradeoff.

    Args:
        embedding_sizes: sequence of EMBEDDING_SIZE values used.
        test_mses: sequence of final test MSE for each size (same order).

    Returns:
        the Axes used.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(embedding_sizes, test_mses, marker='o', color=COLOR_RECON, lw=2)
    ax.set_xscale('log', base=2)
    ax.set_xticks(embedding_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('embedding size')
    ax.set_ylabel('final test MSE')
    ax.set_title('Reconstruction error vs bottleneck size')
    return ax
