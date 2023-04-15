"""Module containing the Gcnn module and its config."""
from typing import Sequence, Tuple

from attrs import define

import torch
from torch import nn


@define
class GcnnConfig:
    """Configuration used for the simple model."""
    input_dimension: int
    layers: Tuple[int]
    activation: bool = True


class GraphConvolutionBlock(nn.Module):
    """A block that performs one graph convolution."""

    def __init__(self, input_dimension: int, output_dimension: int, adjacency_matrices: Sequence[torch.Tensor]):
        """Initializes the convolution block."""
        super().__init__()
        self.adjacency_matrices = [
            torch.nn.functional.normalize(adj_matrix, p=1, dim=1)
            for adj_matrix in adjacency_matrices
        ]
        self.linears = nn.ModuleList([
            nn.Linear(input_dimension, output_dimension)
            for _ in adjacency_matrices
        ])
        self.activation = nn.ReLU()
        self.conv_agg = torch.nn.Conv2d(
            in_channels=len(adjacency_matrices),
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, draft: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            draft: (batch_size, 10, input_dimension) input embeddings per hero

        Returns:
            (batch_size, 10, output_dimension) output embeddings per hero
        """
        input_tensors = [
            torch.matmul(adj_matrix, self.activation(linear(draft)))
            for linear, adj_matrix in zip(self.linears, self.adjacency_matrices)
        ]
        stacked_input = torch.stack(input_tensors, dim=1)
        aggregated = torch.squeeze(self.conv_agg(stacked_input), dim=1)
        return aggregated


class Gcnn(nn.Module):
    """A module for team convolution."""

    def __init__(self, config: GcnnConfig, adjacency_matrices: Sequence[torch.Tensor]):
        """Initialize the Gcnn module with the given config.

        Args:
            config: The config to use for the module
            adjacency_matrices: The matrices determining the graph connections
        """
        super().__init__()
        input_dimensions = [config.input_dimension] + config.layers[:-1]
        output_dimensions = config.layers
        layers = [
            GraphConvolutionBlock(input_dimension=i, output_dimension=o, adjacency_matrices=adjacency_matrices)
            for i, o in zip(input_dimensions, output_dimensions)
        ]
        if config.activation:
            layers.append(nn.ReLU())
        self.sequential = nn.Sequential(*layers)

    def forward(self, draft: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            draft: (batch_size, 10, ...) input tensor for the hero embeddings

        Returns:
            (batch_size, 10, ...) embeddings after convolution
        """
        return self.sequential(draft)
