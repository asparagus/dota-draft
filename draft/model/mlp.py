"""Module containing the MLP module and its config."""
from typing import Tuple

from attrs import define

import torch
from torch import nn


@define
class MLPConfig:
    """Configuration used for the simple model."""
    num_heroes: int
    layers: Tuple[int]


class MLP(nn.Module):
    """A simple model."""

    def __init__(self, config: MLPConfig):
        """Initialize the model with the given config.

        Args:
            config: The config used for this module
        """
        super().__init__()
        embedding_dim = config.layers[0]
        self.embeddings = nn.Embedding(config.num_heroes, embedding_dim, padding_idx=0)
        layers = []
        for i in range(len(config.layers) - 1):
            layers.append(nn.Linear(config.layers[i], config.layers[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config.layers[-1], 1))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Compute the forward pass from the draft up to the logits.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero

        Returns:
            (batch_size, 1 or 2) logits before applying the activation function
        """
        hero_embeddings = self.embeddings(x)

        radiant_hero_embeddings = hero_embeddings.narrow(dim=1, start=0, length=5)
        dire_hero_embeddings = hero_embeddings.narrow(dim=1, start=5, length=5)
        radiant_embedding = radiant_hero_embeddings.sum(dim=1)
        dire_embedding = dire_hero_embeddings.sum(dim=1)
        draft_embedding = radiant_embedding - dire_embedding
        logits = self.sequential(draft_embedding)

        return logits
