"""Module containing the Mlp module and its config."""
from typing import Tuple

from attrs import define

import torch
from torch import nn


@define
class MlpConfig:
    """Configuration used for the simple model."""
    input_dimension: int
    layers: Tuple[int]
    activation: bool = True


class Mlp(nn.Module):
    """A simple model."""

    def __init__(self, config: MlpConfig):
        """Initialize the model with the given config.

        Args:
            config: The config used for this module
        """
        super().__init__()
        layers = [nn.Linear(config.input_dimension, config.layers[0])]
        for i, output_dim in enumerate(config.layers[1:]):
            input_dim = config.layers[i]
            layers.append(nn.ReLU())
            layers.append(nn.Linear(input_dim, output_dim))
        if config.activation:
            layers.append(nn.ReLU())
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            x: (batch_size, ..., config.input_size) input tensor

        Returns:
            (batch_size, ..., config.layers[-1] output of the configured mlps
        """
        return self.sequential(x)
