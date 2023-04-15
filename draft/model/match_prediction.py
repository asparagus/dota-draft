"""Module defining the match prediction task."""
from typing import Dict, Tuple
from attrs import define

import torch
from torch import nn

import torchmetrics

from jigsaw.composite import Composite as JigsawComposite
from jigsaw.piece import Module as JigsawModule
from jigsaw.piece import WrappedLoss

from draft.model.keys import LabelKeys, OutputKeys
from draft.model.modules.mlp import Mlp, MlpConfig
from draft.model.team_modules import TeamMerger, TeamSplitter


@define
class MatchPredictionConfig:
    """Configuration used for the match prediction."""
    symmetric: bool
    mlp_config: MlpConfig


MatchPredictionCrossEntropyLoss = WrappedLoss(
    loss_fn=nn.functional.binary_cross_entropy,
    input_name=OutputKeys.OUTPUT_WIN_PROBABILITIES,
    target_name=LabelKeys.LABEL_RADIANT_WIN,
    name="MatchPredictionCrossEntropyLoss",
)


class MatchPredictionModule(JigsawModule):
    def __init__(self, config: MatchPredictionConfig):
        super().__init__()
        self.key_output_team_embeddings = OutputKeys.OUTPUT_TEAM_EMBEDDINGS
        self.key_output_win_probabilities = OutputKeys.OUTPUT_WIN_PROBABILITIES
        self.mlp = Mlp(config.mlp_config)
        self.config = config

        last_dimension = config.mlp_config.layers[-1]
        self.final_layer = nn.Linear(last_dimension, 1)

        self.splitter = TeamSplitter()
        self.merger = TeamMerger()
        self.symmetric = config.symmetric
        self.activation_fn = nn.Softmax(dim=1)
        if config.symmetric:
            self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=2)
            self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=2)
        else:
            self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
            self.val_accuracy = torchmetrics.classification.BinaryAccuracy()

    def inputs(self) -> Tuple[str]:
        return tuple([self.key_output_team_embeddings])

    def outputs(self) -> Tuple[str]:
        return tuple([self.key_output_win_probabilities])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        team_embeddings = inputs[self.key_output_team_embeddings]
        radiant_team_embeddings, dire_team_embeddings = self.splitter(team_embeddings)
        draft = self.merger(radiant_team_embeddings, dire_team_embeddings)
        draft_mlp_output = self.mlp(draft)
        logits = self.final_layer(draft_mlp_output)

        if self.symmetric:
            flipped_draft = self.merger(dire_team_embeddings, radiant_team_embeddings)
            flipped_draft_output = self.mlp(flipped_draft)
            flipped_logits = self.final_layer(flipped_draft_output)
        else:
            flipped_logits = torch.zeros_like(logits)

        softmax_logits = torch.cat([logits, flipped_logits], dim=1)
        probabilities = self.activation_fn(softmax_logits)[:, 0]

        output = {
            self.key_output_win_probabilities: probabilities
        }
        return output


class MatchPrediction(JigsawComposite):
    def __init__(self, config: MatchPredictionConfig):
        super().__init__([
            MatchPredictionModule(config),
            MatchPredictionCrossEntropyLoss,
        ])
