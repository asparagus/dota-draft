"""Module containing the ModelWrapper lightning module and its config."""
from typing import Dict, Tuple
from attrs import define

import torch
from torch import optim

import pytorch_lightning as pl

from jigsaw.composite import Composite as JigsawComposite
from jigsaw.piece import LossFunction as JigsawLossFunction
from jigsaw.piece import Module as JigsawModule

from draft.model.keys import FeatureKeys, LabelKeys
from draft.model.embedding import Embedding, EmbeddingConfig
from draft.model.team_modules import TeamConvolution, TeamConvolutionConfig
from draft.model.match_prediction import MatchPrediction, MatchPredictionConfig


BATCH_AXIS = 'examples'


@define
class ModelConfig:
    """Configuration used for the model wrapper."""
    embedding_config: EmbeddingConfig
    team_convolution_config: TeamConvolutionConfig
    match_prediction_config: MatchPredictionConfig
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
        self.key_feature_hero_picks = FeatureKeys.FEATURE_HERO_PICKS
        self.key_label_radiant_win = LabelKeys.LABEL_RADIANT_WIN
        self.save_hyperparameters()
        composite = JigsawComposite(
            components=[
                Embedding(config.embedding_config),
                TeamConvolution(config.team_convolution_config),
                MatchPrediction(config.match_prediction_config),
            ]
        )
        self.inner_modules = JigsawComposite(composite.extract(JigsawModule))
        self.losses = torch.jit.ignore(
            JigsawComposite(composite.extract(JigsawLossFunction))
        )
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay

    def preprocess_input(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Preprocessing function for the inputs.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero
        """
        return {self.key_feature_hero_picks: x}

    def preprocess_label(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Preprocessing function for the labels.

        Args:
            y: (batch_size, 1) vector with the results of each match
        """
        return {self.key_label_radiant_win: y.float()}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the model predictions.

        Args:
            x: (batch_size, 10) vector with the IDs for the heroes picked
        """
        data = self.preprocess_input(x)
        return data | self.inner_modules(data)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Run a training step.

        Args:
            batch: A tensor with drafts and outcomes
            batch_idx: The index for this batch
        """
        x, y = batch
        data = self.forward(x)

        losses = self.losses(data | self.preprocess_label(y))
        loss = sum([v for _, v in losses.items()])
        self.log('train_loss', loss, on_epoch=True, on_step=False)

        # self.train_accuracy(out, y)
        # self.log('train_acc', self.train_accuracy, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Run a validation step.

        Args:
            batch: A tensor with drafts and outcomes
            batch_idx: The index for this batch
        """
        x, y = batch
        data = self.forward(x)

        losses = self.losses(data | self.preprocess_label(y))
        loss = sum([v for _, v in losses.items()])
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        # self.val_accuracy(out, y)
        # self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=False)
        return {'loss': loss}

    def configure_optimizers(self):
        """Set up the optimizer."""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def export(self, path: str):
        """Export the model to the given path using ONNX.

        Args:
            path: The path to export the model to
        """
        model_args = torch.ones((1, 10)).int()
        torch.onnx.export(
            self,
            model_args,
            path,
            opset_version=14,
            input_names=self.inner_modules.inputs(),
            output_names=self.inner_modules.outputs(),
            dynamic_axes={
                key: {0: BATCH_AXIS}
                for key in self.inner_modules.inputs() + self.inner_modules.outputs()
            },
        )
