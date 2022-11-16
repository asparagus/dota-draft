"""Module containing the TeamSplitter and TeamMerger modules."""
from typing import Tuple

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
