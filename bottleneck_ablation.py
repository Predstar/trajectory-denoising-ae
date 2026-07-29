"""Sweep EMBEDDING_SIZE, retraining the AE at each size, and plot test MSE vs
embedding size. Justifies the chosen bottleneck size by showing the
compression-quality tradeoff. Slower than a single run -- trains one model
per (embedding size, seed) combination.

Each embedding size is repeated over SEEDS different seeds and reported as
mean +/- std, since single-seed runs can differ substantially due to
initialization/minibatch-order sensitivity (see thesis Section 10).

Run standalone: python bottleneck_ablation.py
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
from visualize import plot_bottleneck_ablation

# --- Hyperparameters (kept small relative to main.py so the sweep finishes
# in reasonable time -- raise these if you have time/compute to spare) ---
N_TRAJECTORIES = 5000
T = 300
NOISE_SIGMA = 0.6
EPOCHS = 40
BATCH_SIZE = 128
TRAIN_SPLIT = 0.6
EMBEDDING_SIZES = [4, 8, 16, 32, 64]
SEEDS = [0, 1, 2]   # repeat each embedding size over these seeds; report mean +/- std
RESULTS_ROOT = "results"


def make_run_dir():
    label = f"bottleneck_ablation_N{N_TRAJECTORIES}_T{T}_sigma{NOISE_SIGMA}_ep{EPOCHS}_seeds{len(SEEDS)}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, f"{timestamp}_{label}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    run_dir = make_run_dir()
    print(f"Saving results to {run_dir}/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data is generated once, with a fixed seed, so all embedding sizes and
    # all seeds train/evaluate on the identical dataset and split; only the
    # model's own initialization/minibatch order varies with SEEDS below.
    np.random.seed(0)
    clean = generate_dataset(n_trajectories=N_TRAJECTORIES, T=T)
    noisy = add_gaussian_noise(clean, sigma=NOISE_SIGMA)
    clean_flat = torch.tensor(clean.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)
    noisy_flat = torch.tensor(noisy.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)

    n_train = int(N_TRAJECTORIES * TRAIN_SPLIT)
    clean_train, clean_test = clean_flat[:n_train], clean_flat[n_train:]
    noisy_train, noisy_test = noisy_flat[:n_train], noisy_flat[n_train:]

    mean_mses = []
    std_mses = []
    all_results = {}
    for embedding_size in EMBEDDING_SIZES:
        seed_mses = []
        for seed in SEEDS:
            print(f"\n--- embedding_size = {embedding_size}, seed = {seed} ---")
            torch.manual_seed(seed)
            model = TrajectoryAE(input_dim=T * 3, embedding_size=embedding_size)
            train_model(model, noisy_train, clean_train, epochs=EPOCHS, batch_size=BATCH_SIZE, device=device)
            test_loss, _ = evaluate_model(model, noisy_test, clean_test, device=device)
            print(f"embedding_size={embedding_size}  seed={seed}  test MSE={test_loss:.6f}")
            seed_mses.append(test_loss)
        all_results[embedding_size] = seed_mses
        mean_mses.append(float(np.mean(seed_mses)))
        std_mses.append(float(np.std(seed_mses)))
        print(f"embedding_size={embedding_size}  mean MSE={mean_mses[-1]:.6f}  std={std_mses[-1]:.6f}")

    plot_bottleneck_ablation(EMBEDDING_SIZES, mean_mses, yerr=std_mses)
    out_path = os.path.join(run_dir, "bottleneck_ablation.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved {out_path}")
    print("Per-seed results:", all_results)
    print("Mean +/- std:", dict(zip(EMBEDDING_SIZES, zip(mean_mses, std_mses))))


if __name__ == "__main__":
    main()
