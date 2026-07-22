"""Feedforward autoencoder for denoising flattened 3D trajectories."""

import torch
from torch import nn


class TrajectoryAE(nn.Module):
    """Fully-connected autoencoder.

    Input/output shape: (batch, T*3) -- a flattened (T, 3) trajectory.
    """

    def __init__(self, input_dim, embedding_size=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, embedding_size),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
