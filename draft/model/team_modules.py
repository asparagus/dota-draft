"""Module containing the TeamSplitter and TeamMerger modules."""
from typing import Dict, Tuple
from attrs import define

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


@define
class TeamConvolutionConfig:
    """Configuration used for the simple model."""
    gcnn_config: GcnnConfig
    teammate_connections: bool
    opponent_connections: bool


class TeamConvolution(JigsawModule):
    """A module for team convolution."""

    SELF_CONNECTIONS = torch.eye(10, dtype=torch.float)
    TEAMMATE_CONNECTIONS = torch.tensor(
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
    )
    OPPONENT_CONNECTIONS = torch.tensor(
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
    )

    def __init__(self, config: TeamConvolutionConfig):
        """Initialize the TeamConvolution module with the given config.

        Args:
            config: The config to use for the module
        """
        super().__init__()
        self.key_output_hero_embeddings = OutputKeys.OUTPUT_HERO_EMBEDDINGS
        self.key_output_team_embeddings = OutputKeys.OUTPUT_TEAM_EMBEDDINGS
        self.config = config
        adjacency_matrices = [self.SELF_CONNECTIONS]
        if config.teammate_connections:
            adjacency_matrices.append(self.TEAMMATE_CONNECTIONS)
        if config.opponent_connections:
            adjacency_matrices.append(self.OPPONENT_CONNECTIONS)
        self.gcnn = Gcnn(config.gcnn_config, adjacency_matrices=adjacency_matrices)

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
