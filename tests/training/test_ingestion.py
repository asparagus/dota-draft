import unittest.mock as um

import numpy as np

from draft.data import match
from draft.training import ingestion


@um.patch('builtins.open', um.mock_open(read_data='{"match_id": 1}\n{"match_id": 2}'))
def test_matches_from_file():
    matches = ingestion.MatchDataset.matches_from_file('file.txt')
    assert len(matches) == 2
    assert matches[0].match_id == 1
    assert matches[1].match_id == 2


def test_draft_from_match():
    data = {
        'match_id': 1,
        'radiant_team': '1,2,3,4,5',
        'dire_team': '6,7,8,9,10',
    }
    m = match.Match(data)
    draft = ingestion.MatchDataset.draft_from_match(m)
    np.testing.assert_array_equal(draft, [1,2,3,4,5,6,7,8,9,10])


def test_numpy_from_match():
    data = {
        'match_id': 1,
        'radiant_team': '1,2,3,4,5',
        'dire_team': '6,7,8,9,10',
        'radiant_win': 1,
    }
    m = match.Match(data)
    draft, result = ingestion.MatchDataset.numpy_from_match(m)
    np.testing.assert_array_equal(draft, [1,2,3,4,5,6,7,8,9,10])
    np.testing.assert_array_equal(result, [1.0])
