"""Module containing the ModelWrapper lightning module and its config."""
from typing import Optional
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
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class ModelWrapper(pl.LightningModule):
    """Wrapper around an ml module."""

    def __init__(self, config: ModelWrapperConfig, module: Optional[torch.nn.Module] = None):
        """Initialize the model with the given config.

        Args:
            config: The config used for this wrapper
            module: The module that does the learning
        """
        super().__init__()
        self.config = config
        self.module = module
        if config.symmetric:
            self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=2)
            self.loss_fn = nn.functional.cross_entropy
            self.activation_fn = nn.Softmax(dim=1)
        else:
            self.accuracy = torchmetrics.classification.BinaryAccuracy()
            self.loss_fn = nn.functional.binary_cross_entropy
            self.activation_fn = nn.Sigmoid()
        self.learning_rate = self.config.learning_rate
        self.weight_decay = self.config.weight_decay

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
        if self.config.symmetric:
            return y.byte()
        return y.float().unsqueeze(1)

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

    def forward(self, x: torch.Tensor):
        """Compute the forward pass from the draft until the win probabilities.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero

        Returns:
            (batch_size, 2) win probabilities for both teams
        """
        processed = self.preprocess_input(x)
        logits = self.module(processed)

        if self.config.symmetric:
            flipped = self.flip(x)
            flipped_processed = self.preprocess_input(flipped)
            flipped_logits = self.module(flipped_processed)
            logits = torch.cat([logits, flipped_logits], dim=1)

        return self.activation_fn(logits)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Run a training step.

        Args:
            batch: A tensor with drafts and outcomes
            batch_idx: The index for this batch
        """
        x, y = batch
        x = self.preprocess_input(x)
        y = self.preprocess_label(y)
        out = self.forward(x)
        loss = self.loss_fn(out, y)
        self.log('train_loss', loss)

        self.accuracy(out, y)
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
        out = self.forward(x)
        loss = self.loss_fn(out, y)
        self.log('val_loss', loss)

        self.accuracy(out, y)
        self.log('val_acc', self.accuracy)
        return {'loss': loss, 'predictions': out}

    def configure_optimizers(self):
        """Set up the optimizer."""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
