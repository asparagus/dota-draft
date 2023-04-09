"""Module containing the TeamSplitter and TeamMerger modules."""
from typing import Dict, Tuple

from attrs import define

import torch
from torch import nn

from jigsaw.piece import Module as JigsawModule

from draft.model.keys import FeatureKeys, OutputKeys


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
    input_dimension: int
    layers: Tuple[int]
    activation: bool = True


class TeamConvolutionBlock(nn.Module):
    """A block that performs one team convolution."""

    def __init__(self, input_dimension: int, output_dimension: int):
        """Initializes the convolution block."""
        super().__init__()
        self.splitter = TeamSplitter()
        self.linear = nn.Linear(input_dimension * 3, output_dimension)

    def forward(self, draft: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            draft: (batch_size, 10, input_dimension) input embeddings per hero

        Returns:
            (batch_size, 10, output_dimension) output embeddings per hero
        """
        radiant, dire = self.splitter(draft)
        radiant_hero_0 = radiant.narrow(dim=1, start=0, length=1)
        radiant_hero_1 = radiant.narrow(dim=1, start=1, length=1)
        radiant_hero_2 = radiant.narrow(dim=1, start=2, length=1)
        radiant_hero_3 = radiant.narrow(dim=1, start=3, length=1)
        radiant_hero_4 = radiant.narrow(dim=1, start=4, length=1)
        dire_hero_0 = dire.narrow(dim=1, start=0, length=1)
        dire_hero_1 = dire.narrow(dim=1, start=1, length=1)
        dire_hero_2 = dire.narrow(dim=1, start=2, length=1)
        dire_hero_3 = dire.narrow(dim=1, start=3, length=1)
        dire_hero_4 = dire.narrow(dim=1, start=4, length=1)

        radiant_sum = (radiant_hero_0 + radiant_hero_1 + radiant_hero_2 + radiant_hero_3 + radiant_hero_4)
        dire_sum = (dire_hero_0 + dire_hero_1 + dire_hero_2 + dire_hero_3 + dire_hero_4)
        radiant_avg = radiant_sum / 5
        dire_avg = dire_sum / 5

        # Input to each computation is the hero, its team (without the hero) and the enemy team
        # Teams aggregations are normalized to maintain the relative scales
        radiant_hero_0_input = torch.cat((radiant_hero_0, (radiant_sum - radiant_hero_0) / 4, dire_avg), dim=2)
        radiant_hero_1_input = torch.cat((radiant_hero_1, (radiant_sum - radiant_hero_1) / 4, dire_avg), dim=2)
        radiant_hero_2_input = torch.cat((radiant_hero_2, (radiant_sum - radiant_hero_2) / 4, dire_avg), dim=2)
        radiant_hero_3_input = torch.cat((radiant_hero_3, (radiant_sum - radiant_hero_3) / 4, dire_avg), dim=2)
        radiant_hero_4_input = torch.cat((radiant_hero_4, (radiant_sum - radiant_hero_4) / 4, dire_avg), dim=2)
        dire_hero_0_input = torch.cat((dire_hero_0, (dire_sum - dire_hero_0) / 4, radiant_avg), dim=2)
        dire_hero_1_input = torch.cat((dire_hero_1, (dire_sum - dire_hero_1) / 4, radiant_avg), dim=2)
        dire_hero_2_input = torch.cat((dire_hero_2, (dire_sum - dire_hero_2) / 4, radiant_avg), dim=2)
        dire_hero_3_input = torch.cat((dire_hero_3, (dire_sum - dire_hero_3) / 4, radiant_avg), dim=2)
        dire_hero_4_input = torch.cat((dire_hero_4, (dire_sum - dire_hero_4) / 4, radiant_avg), dim=2)

        # Mlp is applied to each hero embedding separately
        return torch.cat(
            [
                self.linear(radiant_hero_0_input),
                self.linear(radiant_hero_1_input),
                self.linear(radiant_hero_2_input),
                self.linear(radiant_hero_3_input),
                self.linear(radiant_hero_4_input),
                self.linear(dire_hero_0_input),
                self.linear(dire_hero_1_input),
                self.linear(dire_hero_2_input),
                self.linear(dire_hero_3_input),
                self.linear(dire_hero_4_input),
            ],
            dim=1,
        )


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
        layers = [TeamConvolutionBlock(config.input_dimension, config.layers[0])]
        for i, output_dim in enumerate(config.layers[1:]):
            input_dim = config.layers[i]
            layers.append(nn.ReLU())
            layers.append(TeamConvolutionBlock(input_dim, output_dim))
        if config.activation:
            layers.append(nn.ReLU())
        self.sequential = nn.Sequential(*layers)

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
        team_embeddings = self.sequential(hero_embeddings)
        outputs = {
            self.key_output_team_embeddings: team_embeddings
        }
        return outputs
