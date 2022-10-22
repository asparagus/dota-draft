"""Api object to connect to OpenDota's API.

Implements functions to request and parse OpenDota's data.
API documentation is available at: https://docs.opendota.com/
"""
from typing import Dict, List, Optional

import json
import logging
import os
import requests
from urllib import parse

from draft.data.hero import Hero
from draft.data.match import Match, Matches


class Api(object):

    API_KEY = 'api_key'
    API_URL = 'https://api.opendota.com/api/'
    HEROES_URL = parse.urljoin(API_URL, 'heroes')
    MATCHES_URL = parse.urljoin(API_URL, 'matches/%s')
    PARSED_MATCHES_URL = parse.urljoin(API_URL, 'parsedMatches')
    PUBLIC_MATCHES_URL = parse.urljoin(API_URL, 'publicMatches')

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Api object with a given api_key.

        Args:
            api_key: The key to the opendota api. If omitted, the DOTA_API_KEY
                environment variable will be used.
        """
        self.api_key = api_key or os.getenv('DOTA_API_KEY')
        if self.api_key is None:
            logging.warning('DOTA_API_KEY not set')

    def _request(self, url, *args, **kwargs) -> Dict:
        """Internal request function.

        Args:
            url: The url to request data from
            *args: Arguments to encode into the request
            **kwargs: Arguments to encode into the request
        Returns:
            The parsed JSON.
        """
        params = kwargs.copy()
        params[Api.API_KEY] = self.api_key
        response = requests.get(url, params)
        return json.loads(response.text)

    def heroes(self) -> List[Hero]:
        """Retrieve heroes information."""
        return [Hero(h) for h in self._request(Api.HEROES_URL)]

    def parsed_matches(self, less_than_match_id: Optional[int] = None) -> Matches:
        """Retrieve parsed match ids.

        Args:
            less_than_match_id: (optional) Id to pass to the API call.
                Retrieved match ids will be prior to this id.
        Returns:
            Array of match ids.
        """
        return [
            Match(m)
            for m in self._request(
                Api.PARSED_MATCHES_URL, less_than_match_id=less_than_match_id)
        ]

    def public_matches(self, less_than_match_id: Optional[int] = None) -> Matches:
        """Retrieve public matches.

        Args:
            less_than_match_id: (optional) Id to pass to the API call.
                Retrieved matches will be prior to this id.
        Returns:
            Array of matches.
        """
        return [
            Match(m)
            for m in self._request(
                Api.PUBLIC_MATCHES_URL, less_than_match_id=less_than_match_id)
        ]

    def match(self, match_id: int) -> Match:
        """Retrieve a match.

        Args:
            match_id: The id of the match to retrieve.
        Returns:
            The match object as obtained from the API.
        """
        return Match(self._request(Api.MATCHES_URL % match_id))
