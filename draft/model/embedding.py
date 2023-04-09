"""Module containing the Embedding module and its config."""
from typing import Dict, Tuple
from attrs import define

import torch
from torch import nn

from jigsaw.piece import Module as JigsawModule
from draft.model.keys import FeatureKeys, OutputKeys


@define
class EmbeddingConfig:
    """Configuration for embedding."""
    num_heroes: int
    embedding_size: int


class Embedding(JigsawModule):
    """A simple embedding."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize the model with the given config.

        Args:
            config: The config used for this module
        """
        super().__init__()
        self.key_feature_hero_picks = FeatureKeys.FEATURE_HERO_PICKS
        self.key_output_hero_embedding = OutputKeys.OUTPUT_HERO_EMBEDDINGS
        self.embeddings = nn.Embedding(
            config.num_heroes,
            config.embedding_size,
            padding_idx=0,
        )

    def inputs(self) -> Tuple[str]:
        """Inputs for the embedding module are the hero picks."""
        return tuple([self.key_feature_hero_picks])

    def outputs(self) -> Tuple[str]:
        """Outputs of the embedding module are the hero embeddings."""
        return tuple([self.key_output_hero_embedding])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the forward pass from the draft up to the logits.

        Args:
            x: (batch_size, 10) vector with the IDs for each hero

        Returns:
            (batch_size, 10, config.embedding_size) embeddings
        """
        hero_picks = inputs[self.key_feature_hero_picks]
        outputs = {
            self.key_output_hero_embedding: self.embeddings(hero_picks)
        }
        return outputs
