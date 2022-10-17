"""This module contains the Blob class, which mirrors a GCS Blob."""
import json
import logging
import os
from typing import Optional, Tuple, TYPE_CHECKING

from draft.data import api

if TYPE_CHECKING:
    import google.cloud.storage
    import google.cloud.storage.bucket


class Cache:
    """Cache on GCS used for keeping track of the earliest and latest ids seen."""

    def __init__(self, blob: 'google.cloud.storage.Blob'):
        """Initialize the cache with an instance of a GCS blob.

        Args:
            blob: The blob to be used for the cache.
        """
        self.blob = blob

    @property
    def boundaries(self):
        """Gets the ids that are the boundaries for the data collected.

        Returns a tuple of (earliest, latest) match IDs, where these are None if
        the cache is not set up.
        """
        try:
            data = json.loads(self.blob.download_as_text())
            earliest = data.get('earliest')
            latest = data.get('latest')
            return earliest, latest
        except Exception as e:
            return None, None

    @boundaries.setter
    def boundaries(self, bounds: Tuple[Optional[int], Optional[int]]):
        """Set the boundaries for the data collected.

        Args:
            bounds: The tuple of (earliest, latest) match IDs to save to the GCS blob.
        """
        earliest, latest = bounds
        data = {
            'earliest': earliest,
            'latest': latest,
        }
        payload = json.dumps(data)
        self.blob.upload_from_string(payload)


class Storage:
    """Class handling the storage of match data in GCS."""

    def __init__(
            self,
            bucket: 'google.cloud.storage.bucket.Bucket',
            storage_path: str,
            cache_filename: str,
        ):
        """Initialize the instance with a GCS bucket.

        Args:
            bucket: GCS Bucket instance to use for storage.
            storage_path: Path within the bucket to store the matches.
            cache_filename: File name for the cache file."""
        self.bucket = bucket
        cache_path = os.path.join(storage_path, cache_filename)
        self.cache = Cache(bucket.blob(cache_path))
        self.storage_path = storage_path

    def earliest(self):
        """Gets the earliest match ID that has been stored.

        Requires consulting the cache.
        """
        return self.cache.boundaries[0]

    def latest(self):
        """Gets the latest match ID that has been stored.

        Requires consulting the cache.
        """
        return self.cache.boundaries[1]

    def store(self, batch: 'api.MatchesData'):
        """Store a new batch of matches.

        Args:
            batch: A list of matches to store.
        """
        batch_latest = batch[0][api.MatchID]
        batch_earliest = batch[-1][api.MatchID]
        earliest, latest = self.cache.boundaries
        earliest = min(earliest, batch_earliest) if earliest else batch_earliest
        latest = max(latest, batch_latest) if latest else batch_latest

        path = os.path.join(self.storage_path, '{}.json'.format(batch_latest))
        payload = json.dumps(batch)
        self.bucket.blob(path).upload_from_string(payload)
        self.cache.boundaries = earliest, latest
        logging.info('Stored matches: {}'.format(path))
