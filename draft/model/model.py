"""Module containing the ModelWrapper lightning module and its config."""
from attrs import define

import torch
from torch import nn
from torch import optim

import torchmetrics
import pytorch_lightning as pl

from draft.model.embedding import Embedding, EmbeddingConfig
from draft.model.mlp import Mlp, MlpConfig
from draft.model.team_modules import TeamMerger, TeamSplitter


@define
class ModelConfig:
    """Configuration used for the model wrapper."""
    embedding_config: EmbeddingConfig
    mlp_config: MlpConfig
    symmetric: bool
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class Model(pl.LightningModule):
    """Wrapper around an ml module."""

    def __init__(self, config: ModelConfig):
        """Initialize the model with the given config.

        Args:
            config: The config used for this wrapper
        """
        super().__init__()
        self.save_hyperparameters()
        self.embedding = Embedding(config.embedding_config)
        self.mlp = Mlp(config.mlp_config)
        self.splitter = TeamSplitter()
        self.merger = TeamMerger()
        self.symmetric = config.symmetric
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay

        last_dimension = config.mlp_config.layers[-1]
        self.final_layer = nn.Linear(last_dimension, 1)

        if config.symmetric:
            self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=2)
            self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=2)
            self.loss_fn = nn.functional.cross_entropy
            self.activation_fn = nn.Softmax(dim=1)
        else:
            self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
            self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
            self.loss_fn = nn.functional.binary_cross_entropy
            self.activation_fn = nn.Sigmoid()

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocessing function for the inputs.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero
        """
        return x

    def preprocess_label(self, y: torch.Tensor) -> torch.Tensor:
        """Preprocessing function for the labels.

        Args:
            y: (batch_size, 1) vector with the results of each match
        """
        if self.symmetric:
            return y.byte()
        return y.float().unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass from the draft until the win probabilities.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero

        Returns:
            (batch_size, 2) win probabilities for both teams
        """
        processed = self.preprocess_input(x)
        embeddings = self.embedding(processed)
        radiant, dire = self.splitter(embeddings)
        draft = self.merger(radiant, dire)
        logits = self.final_layer(
            self.mlp(draft)
        )

        if self.symmetric:
            flipped_draft = self.merger(dire, radiant)
            flipped_logits = self.final_layer(
                self.mlp(flipped_draft)
            )
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
        self.log('train_loss', loss, on_epoch=True, on_step=False)

        self.train_accuracy(out, y)
        self.log('train_acc', self.train_accuracy, on_epoch=True, on_step=False)
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
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        self.val_accuracy(out, y)
        self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=False)
        return {'loss': loss, 'predictions': out}

    def configure_optimizers(self):
        """Set up the optimizer."""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
