"""This module contains the Blob class, which mirrors a GCS Blob."""
import json
from functools import cached_property
from typing import Any, Text, TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud import storage


class Blob:
    """
    Class that mirrors a GCS Blob.

    Setting attributes on this class will upload the changes into the blob in GCS.
    Getting attributes from this class will retrieve them from the blob in GCS.
    """

    def __init__(self, gcs_blob: 'storage.Blob'):
        """Initialize the Cache instance with a GCS Blob."""
        object.__setattr__(self, '_gcs_blob', gcs_blob)

    @cached_property
    def _gcs_blob_content(self):
        """Get the underlying GCS blob content loaded as a JSON.
        
        The blob's content will be cached after the first call and only
        invalidated if setting a different value. As such, this function is
        not intended to be thread-safe.
        """
        try:
            return json.loads(self._gcs_blob.download_as_text())
        except Exception as e:
            return {}

    def __getattr__(self, name: Text):
        """Get an attribute from the underlying GCS blob."""
        return self._gcs_blob_content.get(name)

    def __setattr__(self, name: Text, value: Any):
        """Set an attribute and save it to the underlying GCS blob."""
        curr = self._gcs_blob_content
        curr[name] = value
        self._gcs_blob.upload_from_string(json.dumps(curr))
        del self._gcs_blob_content  # Invalidate cache
