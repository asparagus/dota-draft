"""Module containing the TeamSplitter and TeamMerger modules."""
from typing import Dict, Tuple

import torch
from torch import nn

from jigsaw.piece import Module as JigsawModule

from draft.model.keys import OutputKeys
from draft.model.modules.gcnn import GcnnConfig, Gcnn


class TeamSplitter(nn.Module):
    """A module that splits the first 5 and last 5 heroes."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward pass.

        Args:
            x: (batch_size, 10, ...) input tensor

        Returns:
            (batch_size, 5, ...), (batch_size, 5, ...) tensor split down the middle, maybe flipped
        """
        radiant = x.narrow(dim=1, start=0, length=5)
        dire = x.narrow(dim=1, start=5, length=5)
        return radiant, dire


class TeamMerger(nn.Module):
    """A module that merges both team's representations."""

    def forward(self, team_1: torch.Tensor, team_2: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            team_1: (batch_size, 5, ...) input tensor for one team
            team_2: (batch_size, 5, ...) input tensor for the other team

        Returns:
            (batch_size, ...) aggregated tensor merging both teams
        """
        return team_1.sum(dim=1) - team_2.sum(dim=1)


TeamConvolutionConfig = GcnnConfig


class TeamConvolution(JigsawModule):
    """A module for team convolution."""

    def __init__(self, config: TeamConvolutionConfig):
        """Initialize the TeamConvolution module with the given config.

        Args:
            config: The config to use for the module
        """
        super().__init__()
        self.key_output_hero_embeddings = OutputKeys.OUTPUT_HERO_EMBEDDINGS
        self.key_output_team_embeddings = OutputKeys.OUTPUT_TEAM_EMBEDDINGS
        self.config = GcnnConfig
        self.gcnn = Gcnn(
            config,
            adjacency_matrices=[
                # Self-Connections
                torch.eye(10, dtype=torch.float),
                # Teammate-Connections
                torch.tensor(
                    [
                        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    ],
                    dtype=torch.float,
                ),
                # Opponent-Connections
                torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    ],
                    dtype=torch.float,
                ),
            ],
        )

    def inputs(self) -> Tuple[str]:
        """Inputs to the team convolution module are the hero embeddings."""
        return tuple([self.key_output_hero_embeddings])

    def outputs(self) -> Tuple[str]:
        """Outputs from the team convolution module are the team embeddings."""
        return tuple([self.key_output_team_embeddings])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the forward pass.

        Args:
            draft: (batch_size, 10, ...) input tensor for the hero embeddings

        Returns:
            (batch_size, 10, ...) embeddings after convolution
        """
        hero_embeddings = inputs[self.key_output_hero_embeddings]
        outputs = {
            self.key_output_team_embeddings: self.gcnn(hero_embeddings)
        }
        return outputs
