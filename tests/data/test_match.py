from draft.data import match


def test_loads_and_dumps():
    text = '{"match_id": 123, "radiant_win": true, "radiant_team": "1,2,3", "dire_team": "4,5,6"}'
    loaded_match = match.Match.loads(text)
    dumped_text = loaded_match.dumps()
    assert text == dumped_text
    assert loaded_match.match_id == 123
