from attrs import define

import torch
from torch import nn
from torch import optim

import torchmetrics
import pytorch_lightning as pl


@define
class ModelWrapperConfig:
    """Configuration used for the model wrapper."""
    symmetric: bool


class ModelWrapper(pl.LightningModule):
    """Wrapper around an ml module."""

    def __init__(self, config: ModelWrapperConfig, module: torch.nn.Module):
        """Initialize the model with the given config.

        Args:
            config: The config used for this wrapper
            module: The module that does the learning
        """
        super().__init__()
        self.config = config
        self.module = module
        self.accuracy = torchmetrics.Accuracy()
        self.loss_fn = nn.functional.binary_cross_entropy_with_logits
        self.activation_fn = (
            nn.Softmax(dim=1)
            if config.symmetric
            else nn.Sigmoid()
        )
        self.save_hyperparameters(ignore=['module'])

    def preprocess_input(self, x: torch.Tensor):
        """Preprocessing function for the inputs.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero
        """
        return x

    def preprocess_label(self, y: torch.Tensor):
        """Preprocessing function for the labels.

        Args:
            y: (batch_size, 1) vector with the results of each match
        """
        return y

    def flip(self, x: torch.Tensor):
        """Flip teams.

        Used when attempting to ensure symmetric results.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero

        Returns:
            (batch_size, 10) Draft representation with teams flipped
        """
        radiant = x.narrow(dim=1, start=0, length=5)
        dire = x.narrow(dim=1, start=5, length=5)
        return torch.cat([dire, radiant], dim=1)

    def predict_from_logits(self, logits: torch.Tensor):
        return self.activation_fn(logits)[:, 0:1]

    def _forward(self, x: torch.Tensor):
        x = self.preprocess_input(x)
        return self.module(x)

    def forward(self, x: torch.Tensor):
        """Compute the forward pass from the draft until the win probabilities.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero

        Returns:
            (batch_size, 2) win probabilities for both teams
        """
        logits = self._forward(x)

        if self.config.symmetric:
            flipped = self.flip(x)
            flipped_logits = self._forward(flipped)
            logits = torch.cat([logits, flipped_logits], dim=1)

        return self.predict_from_logits(logits)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Run a training step.

        Args:
            batch: A tensor with drafts and outcomes
            batch_idx: The index for this batch
        """
        x, y = batch
        x = self.preprocess_input(x)
        y = self.preprocess_label(y)
        logits = self._forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)

        preds = self.predict_from_logits(logits)
        self.accuracy(preds, y.int())
        self.log('train_acc', self.accuracy)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Run a validation step.

        Args:
            batch: A tensor with drafts and outcomes
            batch_idx: The index for this batch
        """
        x, y = batch
        x = self.preprocess_input(x)
        y = self.preprocess_label(y)
        logits = self._forward(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)

        preds = self.predict_from_logits(logits)
        self.accuracy(preds, y.int())
        self.log('val_acc', self.accuracy)
        return loss

    def configure_optimizers(self):
        """Set up the optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
