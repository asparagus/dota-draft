"""Module that contains the Match class and the Matches type annotation."""
from typing import Dict, List

import json


class Match(dict):
    """Lightweight class for holding and easily accessing match data."""

    MATCH_ID = 'match_id'
    RADIANT_TEAM = 'radiant_team'
    DIRE_TEAM = 'dire_team'
    AVG_RANK_TIER = 'avg_rank_tier'
    RADIANT_WIN = 'radiant_win'
    START_TIME = 'start_time'

    def __init__(self, data: Dict):
        """Initialize the match with a dictionary.

        Args:
            data: Internal data to store
        """
        self._data = data

    @property
    def match_id(self):
        return self._data[Match.MATCH_ID]

    @property
    def radiant_team(self):
        return self._data[Match.RADIANT_TEAM]

    @property
    def dire_team(self):
        return self._data[Match.DIRE_TEAM]

    @property
    def radiant_heroes(self):
        return self.parse_team(self.radiant_team)

    @property
    def dire_heroes(self):
        return self.parse_team(self.dire_team)

    @property
    def avg_rank_tier(self):
        return self._data[Match.AVG_RANK_TIER]

    @property
    def radiant_win(self):
        return self._data[Match.RADIANT_WIN]

    @property
    def start_time(self):
        return self._data[Match.START_TIME]

    def dumps(self):
        """Dump the data to a string."""
        return json.dumps(self._data)

    @classmethod
    def loads(cls, text: str):
        """Load from a string.

        Args:
            text: The text to parse
        """
        return Match(json.loads(text))

    @classmethod
    def parse_team(cls, team: str):
        """Parse a string containing the heroes to a list of ids.

        Args:
            team: String containing the hero ids
        """
        return [int(h) for h in team.split(',')]


# Type annotation
Matches = List[Match]
