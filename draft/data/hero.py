"""Module that contains the Hero class."""
from typing import Dict

import json


class Hero(dict):
    """Lightweight class for holding and easily accessing hero data"""

    ID = 'id'
    NAME = 'name'
    LOCALIZED_NAME = 'localized_name'

    def __init__(self, data: Dict):
        """Initialize the hero with a dictionary.

        Args:
            data: Internal data to store
        """
        self._data = data

    @property
    def _id(self):
        return self._data[Hero.ID]

    @property
    def name(self):
        return self._data[Hero.NAME]

    @property
    def localized_name(self):
        return self._data[Hero.LOCALIZED_NAME]

    def dumps(self):
        """Dump the data to a string."""
        return json.dumps(self._data)

    @classmethod
    def loads(cls, text: str):
        """Load from a string.

        Args:
            text: The text to parse
        """
        return Hero(json.loads(text))
