from typing import Tuple

from attrs import define

import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl


@define
class SimpleModelConfig:
    """Configuration used for the simple model."""
    num_heroes: int
    dimensions: Tuple[int]
    symmetric: bool


class SimpleModel(pl.LightningModule):
    """A simple model."""

    def __init__(self, config: SimpleModelConfig):
        """Initialize the model with the given config.

        Args:
            config: The config used for this run
        """
        super().__init__()
        self.config = config
        embedding_dim = config.dimensions[0]
        self.embeddings = nn.Embedding(config.num_heroes, embedding_dim, padding_idx=0)
        layers = []
        for i in range(len(config.dimensions) - 1):
            layers.append(nn.Linear(config.dimensions[i], config.dimensions[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config.dimensions[-1], 1))
        self.sequential = nn.Sequential(*layers)
        if config.symmetric:
            self.activation = nn.Softmax(dim=1)
            self.loss_fn = nn.functional.binary_cross_entropy_with_logits
        else:
            self.activation = nn.Sigmoid()
            self.loss_fn = nn.functional.cross_entropy
        self.save_hyperparameters()

    def _expand_probabilities(self, p: torch.Tensor):
        """Expand the given probabilities p to (p, 1-p).

        Args:
            p: (batch_size, 1) vector of probabilities.

        Returns:
            (batch_size, 2) vector of probabilities and their complements
        """
        return torch.cat([p, 1 - p], dim=1)

    def _forward(self, x: torch.Tensor):
        """Compute the forward pass from the draft up to the logits.

        If the model is set to compute `symmetric` probabilities, will compute
        the logits for (radiant, dire) and (dire, radiant) and ensure symmetry.

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

        if self.config.symmetric:
            reverse_embedding = dire_embedding - radiant_embedding
            reverse_logits = self.sequential(reverse_embedding)
            logits = torch.cat([logits, reverse_logits], dim=1)

        return logits

    def forward(self, x: torch.Tensor):
        """Compute the forward pass from the draft until the win probabilities.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero

        Returns:
            (batch_size, 2) win probabilities for both teams
        """
        logits = self._forward(x)
        out = self.activation(logits)
        if self.config.symmetric:
            return out
        else:
            return self._expand_probabilities(out)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Run a training step.

        Args:
            batch: A tensor with drafts and outcomes
            batch_idx: The index for this batch
        """
        x, y = batch
        if self.config.symmetric:
            y = self._expand_probabilities(y)
        logits = self._forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Run a validation step.

        Args:
            batch: A tensor with drafts and outcomes
            batch_idx: The index for this batch
        """
        x, y = batch
        if self.config.symmetric:
            y = self._expand_probabilities(y)
        logits = self._forward(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)
        return {'loss': loss}

    def configure_optimizers(self):
        """Set up the optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
