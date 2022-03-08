from unittest import mock

import json

from draft import blob


def test_read():
    blob_content = '{"some_attribute": 23}'
    gcs_blob = mock.Mock()
    gcs_blob.download_as_text = mock.Mock(return_value=blob_content)

    _blob = blob.Blob(gcs_blob)

    assert _blob.some_attribute == 23
    assert _blob.other_attribute is None


def test_write():
    blob_content = '{"some_attribute": 23}'
    gcs_blob = mock.Mock()
    gcs_blob.download_as_text = mock.Mock(return_value=blob_content)
    def modify_blob(text):
        gcs_blob.download_as_text.return_value = text
    gcs_blob.upload_from_string = mock.Mock(side_effect=lambda arg: modify_blob(arg))

    _blob = blob.Blob(gcs_blob)
    _blob.other_attribute = 45
    _blob.another_attribute = 67

    assert gcs_blob.upload_from_string.call_count == 2
    uploaded_text = gcs_blob.upload_from_string.call_args.args[0]
    uploaded_object = json.loads(uploaded_text)
    assert uploaded_object == {
        "some_attribute": 23,
        "other_attribute": 45,
        "another_attribute": 67,
    }


def test_caching():
    gcs_blob = mock.Mock()
    gcs_blob.download_as_text = mock.Mock(return_value='{"some_attribute": 23}')

    _blob = blob.Blob(gcs_blob)
    assert _blob.some_attribute == 23

    # The value is still 23 here because the cache has not been invalidated
    gcs_blob.download_as_text = mock.Mock(return_value='{"some_attribute": 45}')
    assert _blob.some_attribute == 23

    # Dummy update -- invalidates cache
    _blob.invalidate = True
    assert _blob.some_attribute == 45
