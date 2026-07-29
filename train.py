"""Training loop for TrajectoryAE denoising."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, noisy_data, clean_data, epochs=100, batch_size=32, lr=1e-3, device="cpu",
                 val_noisy=None, val_clean=None, patience=30, optimizer_name="adam"):
    """Train the autoencoder to reconstruct clean_data from noisy_data.

    Args:
        model: TrajectoryAE instance.
        noisy_data: torch.Tensor (N, input_dim), noisy input.
        clean_data: torch.Tensor (N, input_dim), denoising target.
        epochs: number of training epochs (max iterations for L-BFGS).
        batch_size: minibatch size (ignored for L-BFGS, which is full-batch).
        lr: learning rate.
        device: 'cpu' or 'cuda'.
        val_noisy, val_clean: optional held-out validation tensors, used for
            early stopping and model selection only -- never for final
            reporting (that is what the separate test split is for).
        patience: stop early if validation loss does not improve for this
            many consecutive epochs (only active when val_noisy/val_clean given).
        optimizer_name: 'adam' (default) or 'lbfgs'. L-BFGS is intended as a
            short full-batch refinement after Adam has already converged, not
            as a replacement for it (see run_lbfgs_refinement below).

    Returns:
        list of per-epoch average train losses, or (train_losses, val_losses)
        if val_noisy/val_clean were provided.
    """
    model.to(device)
    criterion = nn.MSELoss()

    dataset = TensorDataset(noisy_data, clean_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    track_val = val_noisy is not None and val_clean is not None
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if optimizer_name == "lbfgs":
        return _train_lbfgs(model, noisy_data, clean_data, epochs, device,
                             val_noisy, val_clean, track_val)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for noisy_batch, clean_batch in loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()
            recon = model(noisy_batch)
            loss = criterion(recon, clean_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * noisy_batch.size(0)

        epoch_loss /= len(dataset)
        train_losses.append(epoch_loss)

        val_loss = None
        if track_val:
            val_loss, _ = evaluate_model(model, val_noisy, val_clean, device=device)
            val_losses.append(val_loss)
            print(f"Epoch {epoch}/{epochs}  train loss: {epoch_loss:.6f}  val loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{epochs}  loss: {epoch_loss:.6f}")

        if track_val:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}: val loss did not improve "
                      f"for {patience} consecutive epochs (best: {best_val_loss:.6f})")
                break

    if track_val:
        return train_losses, val_losses
    return train_losses


def _train_lbfgs(model, noisy_data, clean_data, max_iter, device, val_noisy, val_clean, track_val):
    """Full-batch L-BFGS training/refinement (Liu & Nocedal, 1989).

    Intended as a short comparison against Adam, not a drop-in replacement:
    L-BFGS is a full-batch, second-order method with very different
    convergence behavior than minibatch Adam.
    """
    model.to(device)
    criterion = nn.MSELoss()
    noisy_data = noisy_data.to(device)
    clean_data = clean_data.to(device)

    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    train_losses = []
    val_losses = []

    def closure():
        optimizer.zero_grad()
        recon = model(noisy_data)
        loss = criterion(recon, clean_data)
        loss.backward()
        train_losses.append(loss.item())
        return loss

    model.train()
    optimizer.step(closure)

    final_train_loss = train_losses[-1] if train_losses else float("nan")
    print(f"L-BFGS finished: {len(train_losses)} closure evaluations, "
          f"final train loss: {final_train_loss:.6f}")

    if track_val:
        val_loss, _ = evaluate_model(model, val_noisy, val_clean, device=device)
        val_losses.append(val_loss)
        print(f"L-BFGS val loss: {val_loss:.6f}")
        return train_losses, val_losses
    return train_losses


def evaluate_model(model, noisy_data, clean_data, device="cpu"):
    """Compute reconstruction loss on held-out data (no gradient updates).

    Returns:
        (loss, reconstructed) tuple.
    """
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        noisy_data = noisy_data.to(device)
        clean_data = clean_data.to(device)
        recon = model(noisy_data)
        loss = criterion(recon, clean_data).item()
    return loss, recon
