"""Sweep NOISE_SIGMA, retraining the AE at each level, and plot test MSE vs sigma.

Characterizes denoising robustness: how reconstruction error degrades as
input noise grows. Slower than a single run -- trains one model per
(sigma, seed) combination.

Each sigma is repeated over SEEDS different seeds and reported as
mean +/- std, since single-seed runs can differ substantially due to
initialization/minibatch-order sensitivity (see thesis Section 10).

Run standalone: python noise_sweep.py
"""

import os
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from trajectory import generate_dataset
from noise import add_gaussian_noise
from model import TrajectoryAE
from train import train_model, evaluate_model
from visualize import plot_noise_sweep

# --- Hyperparameters (kept small relative to main.py so the sweep finishes
# in reasonable time -- raise these if you have time/compute to spare) ---
N_TRAJECTORIES = 800
T = 300
EMBEDDING_SIZE = 8
HIDDEN_DIMS = (128, 64, 32)
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
TRAIN_SPLIT = 0.6
SIGMAS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
SEEDS = [0, 1, 2]   # repeat each sigma over these seeds; report mean +/- std
RESULTS_ROOT = "results"


def make_run_dir():
    hidden_label = "-".join(str(d) for d in HIDDEN_DIMS)
    label = f"noise_sweep_N{N_TRAJECTORIES}_T{T}_emb{EMBEDDING_SIZE}_hid{hidden_label}_ep{EPOCHS}_lr{LR}_seeds{len(SEEDS)}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, f"{timestamp}_{label}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    run_dir = make_run_dir()
    print(f"Saving results to {run_dir}/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Clean data is generated once, with a fixed seed, so all sigmas and all
    # seeds share the same underlying clean trajectories and split; only the
    # noise draw (per sigma) and model init/minibatch order (per seed) vary.
    np.random.seed(0)
    clean = generate_dataset(n_trajectories=N_TRAJECTORIES, T=T)
    clean_flat = torch.tensor(clean.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)
    n_train = int(N_TRAJECTORIES * TRAIN_SPLIT)
    clean_train, clean_test = clean_flat[:n_train], clean_flat[n_train:]

    mean_mses = []
    std_mses = []
    all_results = {}
    for sigma in SIGMAS:
        seed_mses = []
        for seed in SEEDS:
            print(f"\n--- sigma = {sigma}, seed = {seed} ---")
            np.random.seed(seed)
            noisy = add_gaussian_noise(clean, sigma=sigma)
            noisy_flat = torch.tensor(noisy.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)
            noisy_train, noisy_test = noisy_flat[:n_train], noisy_flat[n_train:]

            torch.manual_seed(seed)
            model = TrajectoryAE(input_dim=T * 3, embedding_size=EMBEDDING_SIZE, hidden_dims=HIDDEN_DIMS)
            train_model(model, noisy_train, clean_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=device)
            test_loss, _ = evaluate_model(model, noisy_test, clean_test, device=device)
            print(f"sigma={sigma}  seed={seed}  test MSE={test_loss:.6f}")
            seed_mses.append(test_loss)
        all_results[sigma] = seed_mses
        mean_mses.append(float(np.mean(seed_mses)))
        std_mses.append(float(np.std(seed_mses)))
        print(f"sigma={sigma}  mean MSE={mean_mses[-1]:.6f}  std={std_mses[-1]:.6f}")

    plot_noise_sweep(SIGMAS, mean_mses, yerr=std_mses)
    out_path = os.path.join(run_dir, "noise_sweep.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved {out_path}")
    print("Per-seed results:", all_results)
    print("Mean +/- std:", dict(zip(SIGMAS, zip(mean_mses, std_mses))))


if __name__ == "__main__":
    main()
