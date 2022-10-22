from draft.data import filter
from draft.data import match


def test_valid_match_filter():
    valid_match_filter = filter.ValidMatchFilter()
    valid_match = match.Match({'radiant_team': '1,2,3,4,5', 'dire_team': '6,7,8,9,10'})
    invalid_match = match.Match({'radiant_team': '1,2', 'dire_team': '3,4'})
    assert valid_match_filter(valid_match)
    assert not valid_match_filter(invalid_match)


def test_rank_filter():
    high_rank_filter = filter.HighRankMatchFilter(minimum_rank=50)
    high_rank_match = match.Match({'avg_rank_tier': 80})
    low_rank_match = match.Match({'avg_rank_tier': 10})
    assert high_rank_filter(high_rank_match)
    assert not high_rank_filter(low_rank_match)


def test_filter_negation():
    high_rank_filter = filter.HighRankMatchFilter(minimum_rank=50)
    low_rank_filter = ~high_rank_filter
    high_rank_match = match.Match({'avg_rank_tier': 80})
    low_rank_match = match.Match({'avg_rank_tier': 10})
    assert not low_rank_filter(high_rank_match)
    assert low_rank_filter(low_rank_match)


def test_filter_conjunction():
    high_rank_filter = filter.HighRankMatchFilter(minimum_rank=50)
    low_rank_filter = ~high_rank_filter
    conjunction_filter = high_rank_filter & low_rank_filter
    high_rank_match = match.Match({'avg_rank_tier': 80})
    low_rank_match = match.Match({'avg_rank_tier': 10})
    assert not conjunction_filter(high_rank_match)
    assert not conjunction_filter(low_rank_match)


def test_filter_disjunction():
    high_rank_filter = filter.HighRankMatchFilter(minimum_rank=50)
    low_rank_filter = ~high_rank_filter
    disjunction_filter = high_rank_filter | low_rank_filter
    high_rank_match = match.Match({'avg_rank_tier': 80})
    low_rank_match = match.Match({'avg_rank_tier': 10})
    assert disjunction_filter(high_rank_match)
    assert disjunction_filter(low_rank_match)
