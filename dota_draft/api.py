import json
import os
import requests
from urllib import parse


class Api(object):
    
    API_URL = 'https://api.opendota.com/api/'
    MATCHES_URL = parse.urljoin(API_URL, 'matches/%s')
    PARSED_MATCHES_URL = parse.urljoin(API_URL, 'parsedMatches')

    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv('DOTA_API_KEY')
        self.api_key = api_key
    
    def _request(self, url, *args, **kwargs):
        params = kwargs.copy()
        params['api_key'] = self.api_key
        response = requests.get(url, params)
        return json.loads(response.text)

    def parsed_matches(self, less_than_match_id=None):
        return self._request(
            Api.PARSED_MATCHES_URL, less_than_match_id=less_than_match_id)

    def matches(self, match_id):
        return self._request(Api.MATCHES_URL % match_id)