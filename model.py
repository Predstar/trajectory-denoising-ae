"""Feedforward autoencoder for denoising flattened 3D trajectories."""

from collections.abc import Sequence

import torch
from torch import nn


class TrajectoryAE(nn.Module):
    """Fully-connected autoencoder.

    Input/output shape: (batch, T*3) -- a flattened (T, 3) trajectory.
    """

    def __init__(self, input_dim: int, embedding_size: int = 32,
                 hidden_dims: Sequence[int] = (128,)) -> None:
        """
        Args:
            input_dim: flattened trajectory length (T * 3).
            embedding_size: bottleneck dimension.
            hidden_dims: encoder hidden-layer widths, applied in order from
                input_dim down to embedding_size (e.g. (128, 64) tapers
                input_dim -> 128 -> 64 -> embedding_size). The decoder mirrors
                this sequence in reverse. Pass an empty sequence for a
                single-layer (no hidden layer) autoencoder.
        """
        super().__init__()
        self.encoder = self._build_mlp([input_dim, *hidden_dims, embedding_size],
                                        final_activation=True)
        self.decoder = self._build_mlp([embedding_size, *reversed(hidden_dims), input_dim],
                                        final_activation=False)

    @staticmethod
    def _build_mlp(dims: Sequence[int], final_activation: bool) -> nn.Sequential:
        """Build a stack of Linear+ReLU layers connecting consecutive widths in dims.

        final_activation controls whether a ReLU follows the last Linear layer
        (True for the encoder's output; False for the decoder's, since
        trajectory coordinates can be negative).
        """
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(True))
        if not final_activation:
            layers.pop()
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
