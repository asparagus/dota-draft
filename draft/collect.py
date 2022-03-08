import argparse
import json
import logging
import os

from typing import Callable, Generator, Optional

from google.cloud import storage

from draft import api
from draft import blob


class Storage:

    def __init__(self, bucket, storage_path):
        self.bucket = bucket
        self.storage_path = storage_path

    def store(self, results):
        representative_id = results[0]['match_id']
        path = os.path.join(self.storage_path, '{}.json'.format(representative_id))
        logging.info('Stored results: {}'.format(path))
        self.bucket.blob(path).upload_from_string(json.dumps(results))


class Collector:
    """Class for handling the collection and storage of data from the API."""

    def __init__(
        self,
        api_call: Callable[[Optional[int]], api.MatchesData],
        storage: Storage,
        cache: blob.Blob,
    ):
        """Initialize the instance of Collector.

        Args:
            api_call: Function to call the API and get results.
            storage: Instance used to store data in GCS.
            cache: Instance used to store runtime data in GCS.
        """
        self.api_call = api_call
        self.storage = storage
        self.cache = cache

    def api_slice(
        self,
        start_id: Optional[int] = None,
        stop_id: Optional[int] = None,
    ) -> api.MatchesData:
        """Obtains a slice of data from the API.

        Args:
            start_id: Optional id for the less_than_match_id argument in the API.
            stop_id: Optional id at which to stop retrieving data.

        Returns:
            Arrays of match data retrieved from the API.
            Filters out any match with id <= stop_id.
        """
        results = self.api_call(start_id)
        if results and stop_id is not None:
            last_id = results[-1]['match_id']
            if last_id <= stop_id:
                index = max(i for i, m in enumerate(results)
                            if m['match_id'] > stop_id) + 1
                results = results[:index]
        return results

    def data(
        self,
        start_id: Optional[int] = None,
        stop_id: Optional[int] = None,
    ) -> Generator[api.MatchesData, None, None]:
        """Creates a generator for data from the API.

        Args:
            start_id: Optional id for the less_than_match_id argument in the API.
            stop_id: Optional id at which to stop retrieving data.

        Yields:
            Arrays of matches retrieved from the API.
            Stopping before any match with id <= stop_id.
        """
        current_id = start_id
        results = self.api_slice(start_id=current_id, stop_id=stop_id)
        while results:
            yield results
            current_id = results[-1]['match_id']
            results = self.api_slice(start_id=current_id, stop_id=stop_id)

    def batch(
        self,
        data_generator: Generator[api.MatchesData, None, None],
        batch_size: int,
    ) -> Generator[api.MatchesData, None, None]:
        """Shapes generated data into batches of the given size.

        Args:
            data_generator: Generator for API data.
            batch_size: Size of the batches to generate.

        Yields:
            Arrays of matches retrieved from the API of size batch_size.
        """
        batch = []
        for data in data_generator:
            batch.extend(data)
            while len(batch) >= batch_size:
                yield batch[:batch_size]
                batch = batch[batch_size:]
        if batch:
            yield batch

    def collect(
        self,
        limit: int,
        start_id: Optional[int] = None,
        batch_size: Optional[int] = 1000,
    ):
        """Collect and store new data.

        Args:
            limit: Limit to the number of data collected.
            start_id: Optional id for the less_than_match_id argument in the API.
            batch_size: Size of the batches of data to store.
        """
        earliest_id = self.cache.earliest
        latest_id = self.cache.latest
        stop_id = latest_id if start_id is not None else None

        num_matches = 0
        data_gen = self.data(start_id=start_id, stop_id=stop_id)
        batched_data = self.batch(data_gen, batch_size=batch_size)
        for batch in batched_data:
            self.storage.store(batch)
            latest_new_id = batch[0]['match_id']
            earliest_new_id = batch[-1]['match_id']
            latest_id = max(latest_id or latest_new_id, latest_new_id)
            earliest_id = min(earliest_id or earliest_new_id, earliest_new_id)
            num_matches += len(batch)
            if num_matches >= limit:
                break

        self.cache.latest = latest_id
        self.cache.earliest = earliest_id


def run(start_id, bucket_name, storage_path):
    bucket = storage.Client().bucket(bucket_name)
    api_call = api.Api().public_matches
    _storage = Storage(bucket=bucket, storage_path=storage_path)
    cache_path = os.path.join(storage_path, 'cache.json')
    cache_blob = blob.Blob(bucket.blob(cache_path))
    collector = Collector(api_call=api_call, storage=_storage, cache=cache_blob)
    collector.collect(
        limit=1e6,
        start_id=start_id,
        batch_size=1000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect data from OpenDota API and save to cloud storage bucket')
    parser.add_argument(
        '--start_id',
        help='Match id to determine the backfilling point. Only matches before this will be collected.',
    )
    parser.add_argument(
        '--bucket_name',
        default='dota-draft',
        help='Bucket name to store the results. Default: dota-draft',
    )
    parser.add_argument(
        '--storage_path',
        default='data/matches',
        help='Path within the bucket to store resulting matches. Default: data/matches',
    )
    args = parser.parse_args()
    run(start_id=args.start_id,
        bucket_name=args.bucket_name,
        storage_path=args.storage_path,
    )
