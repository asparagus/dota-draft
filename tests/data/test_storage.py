from unittest import mock
import json
import os
import pytest

from draft.data import storage


def blob_with_content(blob_content):
    gcs_blob = mock.Mock()
    gcs_blob.download_as_text = mock.Mock(return_value=blob_content)
    def modify_blob(text):
        gcs_blob.download_as_text.return_value = text
    gcs_blob.upload_from_string = mock.Mock(side_effect=lambda arg: modify_blob(arg))
    return gcs_blob


@pytest.fixture
def cache_blob():
    return blob_with_content('{"earliest": 1, "latest": 3}')


@pytest.fixture
def empty_blob():
    return blob_with_content('{}')


@pytest.fixture
def missing_blob():
    gcs_blob = mock.Mock()
    gcs_blob.download_as_text = mock.Mock(return_value=Exception('test'))
    def modify_blob(text):
        gcs_blob.download_as_text.return_value = text
    gcs_blob.upload_from_string = mock.Mock(side_effect=lambda arg: modify_blob(arg))
    return gcs_blob


@pytest.fixture
def storage_path():
    return 'matches'


@pytest.fixture
def cache_filename():
    return 'cache.json'


@pytest.fixture
def batch():
    return [
        {'match_id': 8},
        {'match_id': 7},
        {'match_id': 6},
        {'match_id': 5},
    ]


@pytest.fixture
def bucket(cache_blob, empty_blob, storage_path, cache_filename, batch):
    bucket = mock.Mock()
    def blob(path):
        if path == os.path.join(storage_path, cache_filename):
            return cache_blob
        elif path == os.path.join(storage_path, '{}.json'.format(batch[0]['match_id'])):
            return empty_blob
        else:
            return None
    bucket.blob = mock.Mock(side_effect=lambda arg: blob(arg))
    return bucket


def test_cache_read(cache_blob):
    cache = storage.Cache(cache_blob)
    assert cache.boundaries == (1, 3)


def test_cache_write(cache_blob):
    cache = storage.Cache(cache_blob)
    cache.boundaries = (1, 8)

    uploaded_text = cache_blob.upload_from_string.call_args.args[0]
    uploaded_object = json.loads(uploaded_text)
    assert uploaded_object == {
        'earliest': 1,
        'latest': 8,
    }


def test_cache_read_empty(empty_blob):
    cache = storage.Cache(empty_blob)
    assert cache.boundaries == (None, None)


def test_cache_write_empty(empty_blob):
    cache = storage.Cache(empty_blob)
    cache.boundaries = (4, 8)

    uploaded_text = empty_blob.upload_from_string.call_args.args[0]
    uploaded_object = json.loads(uploaded_text)
    assert uploaded_object == {
        'earliest': 4,
        'latest': 8,
    }


def test_cache_read_missing(missing_blob):
    cache = storage.Cache(missing_blob)
    assert cache.boundaries == (None, None)


def test_cache_write_missing(missing_blob):
    cache = storage.Cache(missing_blob)
    cache.boundaries = (4, 8)

    uploaded_text = missing_blob.upload_from_string.call_args.args[0]
    uploaded_object = json.loads(uploaded_text)
    assert uploaded_object == {
        'earliest': 4,
        'latest': 8,
    }


def test_storage_boundaries(bucket, storage_path, cache_filename):
    strg = storage.Storage(bucket=bucket, storage_path=storage_path, cache_filename=cache_filename)
    assert strg.earliest() == 1
    assert strg.latest() == 3


def test_storage_store(bucket, storage_path, cache_filename, batch, cache_blob, empty_blob):
    strg = storage.Storage(bucket=bucket, storage_path=storage_path, cache_filename=cache_filename)
    strg.store(batch)

    # cache_blob is set up to receive the updated cache
    uploaded_cache_text = cache_blob.upload_from_string.call_args.args[0]
    uploaded_cache = json.loads(uploaded_cache_text)
    assert uploaded_cache == {
        'earliest': 1,
        'latest': 8,
    }

    # empty blob is set up to receive the new batch of matches
    uploaded_matches_text = empty_blob.upload_from_string.call_args.args[0]
    uploaded_matches = json.loads(uploaded_matches_text)
    assert uploaded_matches == batch
