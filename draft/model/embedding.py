"""Module containing the Embedding module and its config."""
from attrs import define

import torch
from torch import nn


@define
class EmbeddingConfig:
    """Configuration for embedding."""
    num_heroes: int
    embedding_size: int


class Embedding(nn.Module):
    """A simple model."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize the model with the given config.

        Args:
            config: The config used for this module
        """
        super().__init__()
        self.embeddings = nn.Embedding(
            config.num_heroes,
            config.embedding_size,
            padding_idx=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass from the draft up to the logits.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero

        Returns:
            (batch_size, 10, config.embedding_size) embeddings
        """
        return self.embeddings(x)
