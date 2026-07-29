"""Sweep the AE's hidden layer width (HIDDEN_DIM), retraining at each width,
and plot test MSE / parameter count / runtime vs width. Addresses the
supervisor comment on Section 4.5: "That is already a quite heavy network,
try what happens if you use a leaner AE."

Compares the current default width (128) against leaner widths (64, 32),
holding embedding size and all other hyperparameters fixed. Each width is
repeated over SEEDS different seeds and reported as mean +/- std, since
single-seed runs can differ substantially due to initialization/minibatch
order sensitivity (see thesis Section 10).

Run standalone: python width_ablation.py
"""

import os
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from trajectory import generate_dataset
from noise import add_gaussian_noise
from model import TrajectoryAE
from train import train_model, evaluate_model

# --- Hyperparameters (kept small relative to main.py so the sweep finishes
# in reasonable time -- raise these if you have time/compute to spare) ---
N_TRAJECTORIES = 5000
T = 300
NOISE_SIGMA = 0.6
EMBEDDING_SIZE = 8
EPOCHS = 40
BATCH_SIZE = 128
TRAIN_SPLIT = 0.6
HIDDEN_DIMS = [128, 64, 32]   # 128 = current default; 64/32 = leaner comparisons
SEEDS = [0, 1, 2]              # repeat each width over these seeds; report mean +/- std
RESULTS_ROOT = "results"


def make_run_dir():
    label = f"width_ablation_N{N_TRAJECTORIES}_T{T}_sigma{NOISE_SIGMA}_emb{EMBEDDING_SIZE}_ep{EPOCHS}_seeds{len(SEEDS)}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, f"{timestamp}_{label}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_width_ablation(hidden_dims, mean_mses, std_mses, param_counts, mean_runtimes):
    """Two-panel figure: test MSE vs width, and parameter count vs width,
    both with seed-variability error bars where applicable."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(hidden_dims, mean_mses, yerr=std_mses, marker='o', color="#1baf7a",
                 lw=2, capsize=4, ecolor="#eb6834")
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(hidden_dims)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel('hidden layer width')
    ax1.set_ylabel('final test MSE (mean +/- std over seeds)')
    ax1.set_title('Reconstruction error vs hidden width')

    ax2.plot(hidden_dims, param_counts, marker='o', color="#2a78d6", lw=2)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(hidden_dims)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_xlabel('hidden layer width')
    ax2.set_ylabel('trainable parameters')
    ax2.set_title('Parameter count vs hidden width')

    fig.tight_layout()
    return fig


def main():
    run_dir = make_run_dir()
    print(f"Saving results to {run_dir}/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data is generated once, with a fixed seed, so all widths and all seeds
    # train/evaluate on the identical dataset and split; only the model's own
    # initialization/minibatch order varies with SEEDS below.
    np.random.seed(0)
    clean = generate_dataset(n_trajectories=N_TRAJECTORIES, T=T)
    noisy = add_gaussian_noise(clean, sigma=NOISE_SIGMA)
    clean_flat = torch.tensor(clean.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)
    noisy_flat = torch.tensor(noisy.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)

    n_train = int(N_TRAJECTORIES * TRAIN_SPLIT)
    clean_train, clean_test = clean_flat[:n_train], clean_flat[n_train:]
    noisy_train, noisy_test = noisy_flat[:n_train], noisy_flat[n_train:]

    mean_mses, std_mses, param_counts, mean_runtimes = [], [], [], []
    all_results = {}
    for hidden_dim in HIDDEN_DIMS:
        seed_mses = []
        seed_runtimes = []
        n_params = None
        for seed in SEEDS:
            print(f"\n--- hidden_dim = {hidden_dim}, seed = {seed} ---")
            torch.manual_seed(seed)
            model = TrajectoryAE(input_dim=T * 3, embedding_size=EMBEDDING_SIZE, hidden_dim=hidden_dim)
            n_params = count_params(model)

            start = time.time()
            train_model(model, noisy_train, clean_train, epochs=EPOCHS, batch_size=BATCH_SIZE, device=device)
            elapsed = time.time() - start

            test_loss, _ = evaluate_model(model, noisy_test, clean_test, device=device)
            print(f"hidden_dim={hidden_dim}  seed={seed}  test MSE={test_loss:.6f}  "
                  f"params={n_params:,}  runtime={elapsed:.1f}s")
            seed_mses.append(test_loss)
            seed_runtimes.append(elapsed)

        all_results[hidden_dim] = {"mses": seed_mses, "params": n_params, "runtimes": seed_runtimes}
        mean_mses.append(float(np.mean(seed_mses)))
        std_mses.append(float(np.std(seed_mses)))
        param_counts.append(n_params)
        mean_runtimes.append(float(np.mean(seed_runtimes)))
        print(f"hidden_dim={hidden_dim}  mean MSE={mean_mses[-1]:.6f}  std={std_mses[-1]:.6f}  "
              f"params={n_params:,}  mean runtime={mean_runtimes[-1]:.1f}s")

    plot_width_ablation(HIDDEN_DIMS, mean_mses, std_mses, param_counts, mean_runtimes)
    out_path = os.path.join(run_dir, "width_ablation.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved {out_path}")

    print("\n--- Summary ---")
    print(f"{'width':>8} {'params':>10} {'mean MSE':>12} {'std MSE':>10} {'mean runtime(s)':>16}")
    for hd, mm, sm, pc, rt in zip(HIDDEN_DIMS, mean_mses, std_mses, param_counts, mean_runtimes):
        print(f"{hd:>8} {pc:>10,} {mm:>12.6f} {sm:>10.6f} {rt:>16.1f}")

    print("\nPer-seed results:", all_results)


if __name__ == "__main__":
    main()
