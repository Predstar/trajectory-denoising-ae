"""Sweep EMBEDDING_SIZE, retraining the AE at each size, and plot test MSE vs
embedding size. Justifies the chosen bottleneck size by showing the
compression-quality tradeoff. Slower than a single run -- trains one model
per embedding size.

Run standalone: python bottleneck_ablation.py
"""

import os
from datetime import datetime
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
RESULTS_ROOT = "results"


def make_run_dir():
    label = f"bottleneck_ablation_N{N_TRAJECTORIES}_T{T}_sigma{NOISE_SIGMA}_ep{EPOCHS}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, f"{timestamp}_{label}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    run_dir = make_run_dir()
    print(f"Saving results to {run_dir}/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clean = generate_dataset(n_trajectories=N_TRAJECTORIES, T=T)
    noisy = add_gaussian_noise(clean, sigma=NOISE_SIGMA)
    clean_flat = torch.tensor(clean.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)
    noisy_flat = torch.tensor(noisy.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)

    n_train = int(N_TRAJECTORIES * TRAIN_SPLIT)
    clean_train, clean_test = clean_flat[:n_train], clean_flat[n_train:]
    noisy_train, noisy_test = noisy_flat[:n_train], noisy_flat[n_train:]

    test_mses = []
    for embedding_size in EMBEDDING_SIZES:
        print(f"\n--- embedding_size = {embedding_size} ---")
        model = TrajectoryAE(input_dim=T * 3, embedding_size=embedding_size)
        train_model(model, noisy_train, clean_train, epochs=EPOCHS, batch_size=BATCH_SIZE, device=device)
        test_loss, _ = evaluate_model(model, noisy_test, clean_test, device=device)
        print(f"embedding_size={embedding_size}  final test MSE={test_loss:.6f}")
        test_mses.append(test_loss)

    plot_bottleneck_ablation(EMBEDDING_SIZES, test_mses)
    out_path = os.path.join(run_dir, "bottleneck_ablation.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved {out_path}")
    print(dict(zip(EMBEDDING_SIZES, test_mses)))


if __name__ == "__main__":
    main()
