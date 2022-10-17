from unittest import mock

from draft.data import collect


def test_api_slice_stop_id():
    mock_data = [
        {'match_id': 999},
        {'match_id': 950},
        {'match_id': 800},
        {'match_id': 500},
    ]
    api_call = mock.Mock(return_value=mock_data)
    collector = collect.Collector(api_call, None)
    stop_id = 850
    expected_result = [
        {'match_id': 999},
        {'match_id': 950},
    ]
    assert collector.api_slice(stop_id=stop_id) == expected_result


def test_api_slice_start_id():
    api_call = mock.Mock(return_value=[])
    collector = collect.Collector(api_call, None)
    collector.api_slice(start_id=123)
    assert api_call.called_once_with(123)


def test_data():
    api_results = [
        [
            {'match_id': 999},
            {'match_id': 950},
        ],
        [
            {'match_id': 800},
            {'match_id': 500},
        ],
        [],
    ]
    api_call = mock.Mock(side_effect=api_results)
    collector = collect.Collector(api_call, None)
    data_gen = collector.data(stop_id=600)
    result = list(data_gen)
    expected_result = [
        [
            {'match_id': 999},
            {'match_id': 950},
        ],
        [
            {'match_id': 800},
        ],
    ]
    assert result == expected_result


def test_batch():
    api_results = [
        [
            {'match_id': 999},
            {'match_id': 950},
        ],
        [
            {'match_id': 800},
            {'match_id': 500},
        ],
        [
            {'match_id': 450},
            {'match_id': 400},
        ],
        [],
    ]
    api_call = mock.Mock(side_effect=api_results)
    collector = collect.Collector(api_call, None)
    batches = collector.batch(collector.data(), batch_size=5)
    result = list(batches)
    expected_result = [
        [
            {'match_id': 999},
            {'match_id': 950},
            {'match_id': 800},
            {'match_id': 500},
            {'match_id': 450},
        ],
        [
            {'match_id': 400},
        ],
    ]
    assert result == expected_result