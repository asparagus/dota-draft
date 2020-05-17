"""Api object to connect to OpenDota's API.

Implements functions to request and parse OpenDota's data.
API documentation is available at: https://docs.opendota.com/
"""
import json
import logging
import os
import requests
from urllib import parse


class Api(object):

    API_URL = 'https://api.opendota.com/api/'
    MATCHES_URL = parse.urljoin(API_URL, 'matches/%s')
    PARSED_MATCHES_URL = parse.urljoin(API_URL, 'parsedMatches')

    def __init__(self, api_key=None):
        """Initialize the Api object with a given api_key.

        Args:
            api_key: The key to the opendota api. If omitted, the DOTA_API_KEY
                environment variable will be used.
        """
        self.api_key = api_key or os.getenv('DOTA_API_KEY')
        if self.api_key is None:
            logging.warning('DOTA_API_KEY not set')

    def _request(self, url, *args, **kwargs):
        """Internal request function.

        Args:
            url: The url to request data from
            *args: Arguments to encode into the request
            **kwargs: Arguments to encode into the request
        Returns:
            The parsed JSON.
        """
        params = kwargs.copy()
        params['api_key'] = self.api_key
        response = requests.get(url, params)
        return json.loads(response.text)

    def parsed_matches(self, less_than_match_id=None):
        """Retrieve parsed match ids.

        Args:
            less_than_match_id: (optional) Id to pass to the API call.
                Retrieved match ids will be prior to this id.
        Returns:
            Array of match ids.
        """
        return self._request(
            Api.PARSED_MATCHES_URL, less_than_match_id=less_than_match_id)

    def match(self, match_id):
        """Retrieve a match.

        Args:
            match_id: The id of the match to retrieve.
        Returns:
            The match object as obtained from the API.
        """
        return self._request(Api.MATCHES_URL % match_id)
