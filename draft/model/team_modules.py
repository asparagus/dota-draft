"""Module containing the TeamSplitter and TeamMerger modules."""
from typing import Tuple

from attrs import define

import torch
from torch import nn


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
        team_1_aggregated = team_1.sum(dim=1)
        team_2_aggregated = team_2.sum(dim=1)
        return team_1_aggregated - team_2_aggregated


@define
class TeamConvolutionConfig:
    """Configuration used for the simple model."""
    input_dimension: int
    output_dimension: int
    activation: bool = True


class TeamConvolution(nn.Module):
    """A module for team convolution."""

    def __init__(self, config: TeamConvolutionConfig):
        """Initialize the TeamConvolution module with the given config.

        Args:
            config: The config to use for the module
        """
        super().__init__()
        layers = [nn.Linear(config.input_dimension * 3,config.output_dimension)]
        if config.activation:
            layers.append(nn.ReLU())
        self.sequential = nn.Sequential(*layers)

    def forward(self, team_1: torch.Tensor, team_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward pass.

        Args:
            team_1: (batch_size, 5, ...) input tensor for one team
            team_2: (batch_size, 5, ...) input tensor for the other team

        Returns:
            (batch_size, 5, ...) (batch_size, 5, ...) results for each team
        """
        team_1_aggregated = team_1.sum(dim=1, keepdim=True)
        team_2_aggregated = team_2.sum(dim=1, keepdim=True)
        team_1_single_partials = [team_1.narrow(dim=1, start=i, length=1) for i in range(5)]
        team_2_single_partials = [team_2.narrow(dim=1, start=i, length=1) for i in range(5)]
        team_1_exclusionary_partials = [team_1_aggregated - single_partial for single_partial in team_1_single_partials]
        team_2_exclusionary_partials = [team_2_aggregated - single_partial for single_partial in team_2_single_partials]
        team_1_layer_inputs = [torch.cat((team_1_single_partials[i], team_1_exclusionary_partials[i] / 4, team_2_aggregated / 5), dim=2) for i in range(5)]
        team_2_layer_inputs = [torch.cat((team_2_single_partials[i], team_2_exclusionary_partials[i] / 4, team_1_aggregated / 5), dim=2) for i in range(5)]
        team_1_outputs = [self.sequential(inp) for inp in team_1_layer_inputs]
        team_2_outputs = [self.sequential(inp) for inp in team_2_layer_inputs]
        team_1_final_output = torch.cat(team_1_outputs, dim=1)
        team_2_final_output = torch.cat(team_2_outputs, dim=1)
        return team_1_final_output, team_2_final_output
