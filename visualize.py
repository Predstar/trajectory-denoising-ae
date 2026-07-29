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


def plot_latent_space_pca(embeddings, color_by=None, color_label=None, split_name='test'):
    """Reduce latent embeddings to 2D via PCA and scatter, optionally colored
    by a trajectory parameter (e.g. radius or rise rate).

    Args:
        embeddings: np.ndarray (N, embedding_size).
        color_by: optional np.ndarray (N,) of values to color points by.
        color_label: label for the colorbar, if color_by is given.
        split_name: which data split these embeddings come from (e.g. 'test',
            'train', 'val'), shown in the title.

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
    ax.set_title(f'Latent space (PCA projection) — {split_name} split, N={embeddings.shape[0]} trajectories')
    return ax


def plot_latent_space_pca_overlay(train_embeddings, test_embeddings):
    """Overlay train and test latent embeddings on one PCA projection, for
    spotting a generalization gap during development (e.g. test points
    clustering separately from train points, or scattering more widely).

    Both splits are projected onto components fit jointly on the combined
    embeddings, so a single pair of axes is a fair shared basis for both.

    Args:
        train_embeddings: np.ndarray (N_train, embedding_size).
        test_embeddings: np.ndarray (N_test, embedding_size).

    Returns:
        the Axes used.
    """
    combined = np.concatenate([train_embeddings, test_embeddings], axis=0)
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T

    n_train = train_embeddings.shape[0]
    proj_train, proj_test = proj[:n_train], proj[n_train:]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(proj_train[:, 0], proj_train[:, 1], color=COLOR_CLEAN, s=10,
               alpha=0.6, label=f'train (N={n_train})')
    ax.scatter(proj_test[:, 0], proj_test[:, 1], color=COLOR_NOISY, s=14,
               alpha=0.8, marker='^', label=f'test (N={test_embeddings.shape[0]})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Latent space (PCA projection) — train vs test overlay')
    ax.legend()
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


def plot_snr_improvement(clean, noisy, reconstructed, n_samples=30, seed=0, highlight_idx=None):
    """Bar chart comparing per-sample SNR before (noisy) vs after (reconstructed)
    denoising, for a random subset of trajectories.

    SNR (dB) = 10 * log10(signal_power / noise_power), where signal_power is
    the clean trajectory's power and noise_power is the MSE against clean.

    Args:
        clean, noisy, reconstructed: np.ndarray (N, T, 3) or (N, T*3).
        highlight_idx: optional trajectory index (e.g. the worst-case test
            trajectory) to force into the sampled set and mark distinctly,
            so it is guaranteed visible even if the random sample would
            otherwise have missed it.

    Returns:
        the Axes used.
    """
    clean = clean.reshape(clean.shape[0], -1)
    noisy = noisy.reshape(noisy.shape[0], -1)
    reconstructed = reconstructed.reshape(reconstructed.shape[0], -1)

    n = clean.shape[0]
    rng = np.random.default_rng(seed)
    sample_size = min(n_samples, n) - (1 if highlight_idx is not None else 0)
    pool = np.delete(np.arange(n), highlight_idx) if highlight_idx is not None else np.arange(n)
    idx = rng.choice(pool, size=min(sample_size, len(pool)), replace=False)
    if highlight_idx is not None:
        idx = np.append(idx, highlight_idx)
    idx.sort()

    signal_power = (clean[idx] ** 2).mean(axis=1)
    noise_power_before = ((noisy[idx] - clean[idx]) ** 2).mean(axis=1)
    noise_power_after = ((reconstructed[idx] - clean[idx]) ** 2).mean(axis=1)

    snr_before = 10 * np.log10(signal_power / noise_power_before)
    snr_after = 10 * np.log10(signal_power / noise_power_after)

    x = np.arange(len(idx))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    before_colors = [COLOR_NOISY] * len(idx)
    after_colors = [COLOR_RECON] * len(idx)
    edge_colors_before = ["none"] * len(idx)
    edge_colors_after = ["none"] * len(idx)
    if highlight_idx is not None:
        pos = int(np.where(idx == highlight_idx)[0][0])
        edge_colors_before[pos] = "black"
        edge_colors_after[pos] = "black"
    ax.bar(x - width / 2, snr_before, width, label='before (noisy)', color=before_colors,
           edgecolor=edge_colors_before, linewidth=2)
    ax.bar(x + width / 2, snr_after, width, label='after (reconstructed)', color=after_colors,
           edgecolor=edge_colors_after, linewidth=2)
    tick_labels = [f"{i}*" if i == highlight_idx else str(i) for i in idx]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=90)
    ax.set_xlabel('test trajectory index' + ('  (* = worst-case, outlined)' if highlight_idx is not None else ''))
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR improvement from denoising')
    ax.legend()
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


def plot_noise_sweep(sigmas, test_mses, yerr=None, breakdown_sigma=None):
    """Plot test MSE vs noise level (sigma), characterizing model robustness.

    Args:
        sigmas: sequence of NOISE_SIGMA values used.
        test_mses: sequence of (mean) final test MSE for each sigma (same order).
        yerr: optional sequence of std devs across seeds, for error bars.
        breakdown_sigma: optional sigma value; the plot region from this sigma
            to the end is shaded red and labeled as where reconstruction
            error grows sharply, to flag the point beyond which denoising
            quality degrades fastest.

    Returns:
        the Axes used.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    err_label = 'std across seeds (3 models trained per sigma)' if yerr is not None else None
    ax.errorbar(sigmas, test_mses, marker='o', color=COLOR_CLEAN, lw=2, markersize=7,
                label='mean test MSE (blue line = model reconstruction error)')
    if yerr is not None:
        ax.errorbar(sigmas, test_mses, yerr=yerr, fmt='none', ecolor=COLOR_NOISY,
                     capsize=4, elinewidth=1.5, label=err_label)

    if breakdown_sigma is not None and breakdown_sigma in sigmas:
        idx = sigmas.index(breakdown_sigma)
        ax.axvspan(sigmas[idx], sigmas[-1], color='red', alpha=0.08, zorder=0)
        ax.axvline(sigmas[idx], color='red', linestyle=':', lw=1.5)
        ax.annotate('sharp degradation\nbeyond this point',
                     xy=(sigmas[idx], test_mses[idx]), xytext=(0, 30),
                     textcoords='offset points', ha='center', color='red',
                     fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('noise sigma (input noise std dev)')
    ax.set_ylabel('final test MSE' + (' (mean +/- std over seeds)' if yerr is not None else ''))
    ax.set_title('Reconstruction error vs noise level')
    ax.legend(loc='upper left', fontsize=8)
    return ax


def _unused():
    pass


def plot_bottleneck_ablation(embedding_sizes, test_mses, yerr=None):
    """Plot test MSE vs embedding (bottleneck) size, showing the
    compression-quality tradeoff.

    Args:
        embedding_sizes: sequence of EMBEDDING_SIZE values used.
        test_mses: sequence of (mean) final test MSE for each size (same order).
        yerr: optional sequence of std devs across seeds, for error bars.

    Returns:
        the Axes used.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(embedding_sizes, test_mses, yerr=yerr, marker='o', color=COLOR_RECON, lw=2,
                capsize=4, ecolor=COLOR_NOISY)
    ax.set_xscale('log', base=2)
    ax.set_xticks(embedding_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('embedding size')
    ax.set_ylabel('final test MSE' + (' (mean +/- std over seeds)' if yerr is not None else ''))
    ax.set_title('Reconstruction error vs bottleneck size')
    return ax
