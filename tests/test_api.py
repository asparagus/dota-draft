import os

from draft import api


def test_init_without_key():
    os.environ['DOTA_API_KEY'] = '123'
    a = api.Api()
    assert a.api_key == '123', 'Failed to retrieve key from environment'


def test_init_with_key():
    a = api.Api('456')
    assert a.api_key == '456', 'Failed to save key from constructor'


def test_parsed_matches():
    a = api.Api()
    matches = a.parsed_matches()
    assert isinstance(matches, list), 'Failed to retrieve list of matches'


def test_parsed_matches_with_argument():
    a = api.Api()
    matches = a.parsed_matches()
    last_id = matches[-1]['match_id']

    next_matches = a.parsed_matches(less_than_match_id=last_id)
    largest_id = max([m['match_id'] for m in next_matches])
    assert largest_id < last_id


def test_match():
    a = api.Api()
    matches = a.parsed_matches()
    first_id = matches[0]['match_id']
    match = a.match(first_id)

    assert match is not None, 'Failed to retrieve match'
    for field in ('match_id', 'lobby_type', 'game_mode',
                  'duration', 'radiant_win', 'picks_bans', 'players'):
        assert field in match, 'Match data missing %s' % field
