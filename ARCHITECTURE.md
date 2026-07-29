# Architecture

## Pipeline Overview

```
trajectory.py          noise.py              model.py              train.py
  generate spiral  -->   add Gaussian    -->   TrajectoryAE    -->   train_model /
  trajectories           noise                 (encoder/decoder)     evaluate_model
        |                                            ^                     |
        v                                            |                     v
   data/*.npy  ------------------------------- main.py orchestrates ---> results/<run>/
                                                       |
                                                       v
                                              visualize.py (plots)
                                              animate_preview.py (mp4)
```

`main.py` is the single entry point that wires every module together for one
full training run: generate data -> add noise -> build model -> train ->
evaluate -> save artifacts + plots. `bottleneck_ablation.py` and
`noise_sweep.py` are standalone scripts that reuse the same building blocks
to sweep one hyperparameter at a time.

## Module Responsibilities

| File | Responsibility |
|---|---|
| `main.py` | Orchestration: generates data, builds/trains/evaluates the model, saves every plot and artifact into a timestamped `results/` subfolder. |
| `trajectory.py` | Synthetic 3D spiral ("circular motion") trajectory generation. |
| `noise.py` | Additive i.i.d. Gaussian noise injection. |
| `model.py` | `TrajectoryAE` — the fully-connected denoising autoencoder (`nn.Module`). |
| `train.py` | Training loop (`train_model`) with early stopping, and evaluation (`evaluate_model`). |
| `visualize.py` | All static plotting functions (loss curves, reconstruction overlays, residuals, MSE histogram, SNR bars, PCA of latent space, ablation plots). |
| `animate_preview.py` | Renders one sample trajectory (clean/noisy/reconstructed) as an MP4 animation. |
| `bottleneck_ablation.py` | Standalone sweep of `EMBEDDING_SIZE`, retraining the model at each size, to characterize the capacity/reconstruction-quality tradeoff. |
| `noise_sweep.py` | Standalone sweep of `NOISE_SIGMA`, retraining the model at each level, to characterize noise robustness. |
| `inspect_data.py` | CLI utility to summarize/preview/export saved `data/*.npy` datasets. |

## Data Representation

A single trajectory is a matrix $\tau \in \mathbb{R}^{T \times 3}$ (timesteps
x [x, y, z]). The full dataset is a tensor
$X \in \mathbb{R}^{N \times T \times 3}$. Because `TrajectoryAE` is a
fully-connected network, each trajectory is flattened to a vector
$x \in \mathbb{R}^{T \cdot 3}$ before being fed in (`main.py`,
`clean.reshape(N_TRAJECTORIES, -1)`).

## Spiral Generation (`trajectory.py`)

Each trajectory is parameterized by 4 randomized scalars — starting radius
$r_0$, radius growth rate $k$, vertical speed $v_z$, and total arc
$t_{\max}$:

```
r(t) = r0 + k*t
x(t) = r(t) * cos(t)
y(t) = r(t) * sin(t)
z(t) = vz * t
```

This gives the trajectory family an intrinsic dimensionality of 4, regardless
of how many timesteps `T` it's sampled at — which is why a small autoencoder
bottleneck (`EMBEDDING_SIZE`) is sufficient to capture the underlying shape.

## Noise Model (`noise.py`)

Independent Gaussian noise $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ is
added to every $(x, y, z)$ coordinate at every timestep, controlled by
`NOISE_SIGMA`.

## Model (`model.py`)

`TrajectoryAE` is a symmetric fully-connected autoencoder:

```
Encoder: Linear(input_dim -> 128) -> ReLU -> Linear(128 -> embedding_size) -> ReLU
Decoder: Linear(embedding_size -> 128) -> ReLU -> Linear(128 -> input_dim)
```

- `input_dim = T * 3` (flattened trajectory length).
- `embedding_size` is the bottleneck dimension — the key hyperparameter
  governing the capacity/compression tradeoff (see
  `AE_analysis.pdf` / `AE_Mathematical_Foundations.pdf` for the underlying
  math and experimental comparison).
- No activation is applied after the final decoder layer, since trajectory
  coordinates can be negative.
- `model.encode(x)` / `model.decode(z)` expose the encoder and decoder
  independently (used for latent-space inspection in `visualize.py`).

## Training (`train.py`)

- Optimizer: Adam (`lr` configurable, default `1e-3`).
- Loss: MSE between the reconstructed and clean trajectory.
- `train_model(...)` supports an optional held-out test set
  (`test_noisy`/`test_clean`), evaluating test loss every epoch.
- **Early stopping**: training halts if test loss does not improve for
  `patience` (default 30) consecutive epochs, restoring the run to whichever
  epoch's weights are current at that point and printing the stopping epoch.

## Orchestration (`main.py`)

For each run, `main.py`:

1. Generates a clean/noisy dataset pair and saves it to `data/`.
2. Splits into train/test sets by `TRAIN_SPLIT`.
3. Builds `TrajectoryAE`, prints a `torchinfo` layer-by-layer summary.
4. Trains via `train_model` (with early stopping).
5. Evaluates on the held-out test set, saves:
   - `reconstructed_trajectories.npy` — denoised test-set output.
6. Generates and saves all evaluation plots via `visualize.py` (loss curve,
   reconstruction overlays, per-axis time series, residuals, MSE histogram,
   SNR improvement, latent-space PCA).
7. Renders an MP4 animation of one sample trajectory via
   `animate_preview.py`.

All artifacts for a run are written to a single timestamped, hyperparameter-
labelled folder under `results/`, e.g.
`results/2026-07-22_21-30-15_N1000_T800_sigma0.6_emb32_ep400_lr0.001/`.

## Ablation Scripts

Both ablation scripts reuse `trajectory.py` / `noise.py` / `model.py` /
`train.py` directly (they do not go through `main.py`), training one model
per swept value and plotting test MSE against it:

- `bottleneck_ablation.py` sweeps `EMBEDDING_SIZE` (default
  `[4, 8, 16, 32, 64]`) at fixed noise level.
- `noise_sweep.py` sweeps `NOISE_SIGMA` (default
  `[0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]`) at fixed bottleneck size.

Both use a smaller `N_TRAJECTORIES` / `T` / `EPOCHS` than `main.py`'s default
so the sweep completes in reasonable time (they train one full model per
swept value).
