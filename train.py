"""Training loop for TrajectoryAE denoising."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, noisy_data, clean_data, epochs=100, batch_size=32, lr=1e-3, device="cpu",
                 test_noisy=None, test_clean=None, patience=30):
    """Train the autoencoder to reconstruct clean_data from noisy_data.

    Args:
        model: TrajectoryAE instance.
        noisy_data: torch.Tensor (N, input_dim), noisy input.
        clean_data: torch.Tensor (N, input_dim), denoising target.
        epochs: number of training epochs.
        batch_size: minibatch size.
        lr: learning rate.
        device: 'cpu' or 'cuda'.
        test_noisy, test_clean: optional held-out tensors. If given, test loss
            is computed after every epoch (in addition to train loss).
        patience: stop early if test loss does not improve for this many
            consecutive epochs (only active when test_noisy/test_clean given).

    Returns:
        list of per-epoch average train losses, or (train_losses, test_losses)
        if test_noisy/test_clean were provided.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(noisy_data, clean_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    track_test = test_noisy is not None and test_clean is not None
    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    epochs_without_improvement = 0

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

        test_loss = None
        if track_test:
            test_loss, _ = evaluate_model(model, test_noisy, test_clean, device=device)
            test_losses.append(test_loss)
            print(f"Epoch {epoch}/{epochs}  train loss: {epoch_loss:.6f}  test loss: {test_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{epochs}  loss: {epoch_loss:.6f}")

        if track_test:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}: test loss did not improve "
                      f"for {patience} consecutive epochs (best: {best_test_loss:.6f})")
                break

    if track_test:
        return train_losses, test_losses
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
