"""Orchestration: generate data -> add noise -> train AE -> evaluate -> plot."""

import os
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchinfo import summary
from torchviz import make_dot

from trajectory import generate_dataset
from noise import add_gaussian_noise
from model import TrajectoryAE
from train import train_model, evaluate_model
from visualize import (
    plot_clean_noisy_reconstructed, plot_training_curve,
    plot_per_axis_timeseries, plot_residuals, plot_latent_space_pca,
    plot_mse_histogram, plot_snr_improvement, plot_trajectory_overlay,
)
from animate_preview import animate_sample

# torchviz shells out to Graphviz's `dot` binary; on Windows the installer
# doesn't always update PATH for already-open shells, so add the default
# install location as a fallback.
_default_graphviz_bin = r"C:\Program Files\Graphviz\bin"
if os.path.isdir(_default_graphviz_bin) and _default_graphviz_bin not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + _default_graphviz_bin

# --- Hyperparameters ---
N_TRAJECTORIES = 1000   # total number of synthetic spiral trajectories to generate
T = 800                # timesteps per trajectory
NOISE_SIGMA = 1.6      # std dev of Gaussian noise added to each (x, y, z) point
EMBEDDING_SIZE = 8   # size of the AE's compressed latent representation
EPOCHS = 400           # number of full passes over the training set
BATCH_SIZE = 64       # trajectories per gradient update
LR = 1e-3               # Adam learning rate (try 1e-4 to 1e-2 if loss stalls or diverges)
TRAIN_SPLIT = 0.6      # fraction of trajectories used for training vs testing
DATA_DIR = "data"       # folder where generated datasets are saved
RESULTS_ROOT = "results"  # parent folder for all per-run result subfolders


def make_run_dir():
    """Create a fresh timestamped, labelled subfolder under RESULTS_ROOT for
    this run's plots, e.g. results/2026-07-22_18-30-05_N10_T650_sigma0.2_emb8/.
    """
    label = (
        f"N{N_TRAJECTORIES}_T{T}_sigma{NOISE_SIGMA}_emb{EMBEDDING_SIZE}"
        f"_ep{EPOCHS}_lr{LR}"
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, f"{timestamp}_{label}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    run_dir = make_run_dir()
    print(f"Saving results to {run_dir}/")

    def save(name):
        path = os.path.join(run_dir, name)
        plt.savefig(path, dpi=120)
        return path

    # 1. Generate ground-truth spiral trajectories and their noisy counterparts.
    #    clean/noisy shape: (N_TRAJECTORIES, T, 3)
    clean = generate_dataset(n_trajectories=N_TRAJECTORIES, T=T)
    noisy = add_gaussian_noise(clean, sigma=NOISE_SIGMA)

    # 1b. Save the generated dataset to disk so it can be reused/inspected later.
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "clean_trajectories.npy"), clean)
    np.save(os.path.join(DATA_DIR, "noisy_trajectories.npy"), noisy)
    print(f"Saved datasets to {DATA_DIR}/clean_trajectories.npy and {DATA_DIR}/noisy_trajectories.npy")

    # 1c. Overlay a sample of clean trajectories to show the dataset's spread.
    plot_trajectory_overlay(clean, seed=0)
    save("trajectory_overlay.png")

    # 2. Flatten each (T, 3) trajectory into a single (T*3,) vector so it can
    #    be fed into the fully-connected autoencoder, and convert to tensors.
    clean_flat = torch.tensor(clean.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)
    noisy_flat = torch.tensor(noisy.reshape(N_TRAJECTORIES, -1), dtype=torch.float32)

    # 3. Split into train/test sets (first n_train samples train, rest test).
    n_train = int(N_TRAJECTORIES * TRAIN_SPLIT)
    clean_train, clean_test = clean_flat[:n_train], clean_flat[n_train:]
    noisy_train, noisy_test = noisy_flat[:n_train], noisy_flat[n_train:]

    # 4. Build the autoencoder: input/output dim = T*3 (flattened trajectory),
    #    bottleneck dim = EMBEDDING_SIZE.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrajectoryAE(input_dim=T * 3, embedding_size=EMBEDDING_SIZE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Using device: {device}")
    print(f"Training samples : {n_train}")
    print(f"Test samples     : {N_TRAJECTORIES - n_train}")
    print(model)
    print(f"\nTotal trainable parameters: {n_params:,}")

    # 4b. torchinfo layer-by-layer summary (input/output shapes, param counts).
    #     verbose=0 + manual print avoids UnicodeEncodeError on Windows consoles
    #     (cp1252) that can't render torchinfo's box-drawing characters.
    model_summary = summary(model, input_size=(BATCH_SIZE, T * 3), device=device, verbose=0)
    print(str(model_summary).encode("ascii", "replace").decode("ascii"))

    print("\nStarting training...\n")

    # 5. Train the AE to map noisy trajectories -> clean trajectories (denoising),
    #    tracking test loss every epoch alongside train loss.
    train_losses, test_losses = train_model(
        model, noisy_train, clean_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        device=device, test_noisy=noisy_test, test_clean=clean_test)

    # 6. Evaluate final reconstruction quality on held-out (unseen) trajectories.
    test_loss, recon = evaluate_model(model, noisy_test, clean_test, device=device)
    print(f"Final test loss: {test_loss:.6f}")

    # 6a. TorchViz computation graph of the forward pass.
    sample_input = noisy_test[:1].to(device).requires_grad_(True)
    sample_output = model(sample_input)
    dot = make_dot(sample_output, params=dict(model.named_parameters()))
    dot.format = "png"
    dot.render(os.path.join(run_dir, "computation_graph"), cleanup=True)
    print(f"Saved computation_graph.png to {run_dir}/")

    # 6b. Save the trained model as a traced module so Netron can render the
    #     full architecture graph (a bare state_dict only shows tensor names).
    model_path = os.path.join(run_dir, "model.pt")
    model.eval()
    traced = torch.jit.trace(model, noisy_test[:1].to(device))
    traced.save(model_path)
    print(f"Saved trained model to {model_path}")

    # 6b. Save the denoised (reconstructed) test-set output to disk, shaped like
    #     the clean/noisy datasets: (n_test, T, 3).
    n_test = N_TRAJECTORIES - n_train
    recon_array = recon.cpu().numpy().reshape(n_test, T, 3)
    np.save(os.path.join(DATA_DIR, "reconstructed_trajectories.npy"), recon_array)
    print(f"Saved denoised output to {DATA_DIR}/reconstructed_trajectories.npy")

    # 7. Plot the training/test loss curve and save it.
    plot_training_curve(train_losses, test_losses)
    save("loss_curve.png")

    # 8. Pick one test trajectory, visualize clean vs noisy vs reconstructed, and save it.
    #    Also pick the worst-reconstructed test trajectory to inspect a bad case.
    n_test = N_TRAJECTORIES - n_train
    per_traj_mse = ((recon.cpu().numpy() - clean_test.numpy()) ** 2).reshape(n_test, -1).mean(axis=1)
    worst_idx = int(per_traj_mse.argmax())

    sample_idx = 0
    clean_sample = clean_test[sample_idx].numpy().reshape(T, 3)
    noisy_sample = noisy_test[sample_idx].numpy().reshape(T, 3)
    recon_sample = recon[sample_idx].cpu().numpy().reshape(T, 3)

    plot_clean_noisy_reconstructed(clean_sample, noisy_sample, recon_sample)
    save("reconstruction.png")

    clean_worst = clean_test[worst_idx].numpy().reshape(T, 3)
    noisy_worst = noisy_test[worst_idx].numpy().reshape(T, 3)
    recon_worst = recon[worst_idx].cpu().numpy().reshape(T, 3)
    plot_clean_noisy_reconstructed(clean_worst, noisy_worst, recon_worst)
    save("reconstruction_worst.png")

    print("Saved loss_curve.png, reconstruction.png, reconstruction_worst.png")

    # 8b. Per-axis time series and residual plots for the same sample.
    plot_per_axis_timeseries(clean_sample, noisy_sample, recon_sample)
    save("per_axis_timeseries.png")

    plot_residuals(clean_sample, recon_sample)
    save("residuals.png")

    # 8c. MSE distribution across the whole test set.
    plot_mse_histogram(clean_test.numpy(), recon.cpu().numpy())
    save("mse_histogram.png")

    # 8d. SNR improvement bar chart for a random subset of test trajectories.
    plot_snr_improvement(clean_test.numpy(), noisy_test.numpy(), recon.cpu().numpy())
    save("snr_improvement.png")

    # 8e. Latent space visualization, colored by mean trajectory radius.
    with torch.no_grad():
        test_embeddings = model.encode(noisy_test.to(device)).cpu().numpy()
    clean_test_np = clean_test.numpy().reshape(n_test, T, 3)
    mean_radius = np.sqrt(clean_test_np[:, :, 0] ** 2 + clean_test_np[:, :, 1] ** 2).mean(axis=1)
    plot_latent_space_pca(test_embeddings, color_by=mean_radius, color_label='mean radius')
    save("latent_space_pca.png")

    print("Saved per_axis_timeseries.png, residuals.png, mse_histogram.png, "
          "snr_improvement.png, latent_space_pca.png")

    # 9. Animate the same sample trajectory (clean vs noisy vs reconstructed) as an MP4,
    #    into the run folder. sample_idx indexes the full dataset for clean/noisy;
    #    recon_idx indexes reconstructed_trajectories.npy, which only covers the test split.
    anim_path = os.path.join(run_dir, "trajectory_anim.mp4")
    animate_sample(data_dir=DATA_DIR, sample_idx=n_train + sample_idx, recon_idx=sample_idx,
                   out_path=anim_path)

    # 10. Open all generated plots/animation with the OS default viewer.
    for name in ("trajectory_overlay.png", "loss_curve.png", "reconstruction.png",
                 "reconstruction_worst.png", "per_axis_timeseries.png", "residuals.png",
                 "mse_histogram.png", "snr_improvement.png", "latent_space_pca.png",
                 "trajectory_anim.mp4"):
        os.startfile(os.path.join(run_dir, name))


if __name__ == "__main__":
    main()
