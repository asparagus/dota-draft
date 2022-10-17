"""Cloud Function scheduled to retrieve new data from the API.""" 
import argparse

from typing import Callable, Generator, Optional

import google.cloud.storage

from draft.data import api
from draft.data import storage


class Collector:
    """Class for handling the collection and storage of data from the API."""

    def __init__(
        self,
        api_call: Callable[[Optional[int]], api.MatchesData],
        storage: storage.Storage,
    ):
        """Initialize the instance of Collector.

        Args:
            api_call: Function to call the API and get results.
            storage: Instance used to store data in GCS.
        """
        self.api_call = api_call
        self.storage = storage

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
            last_id = results[-1][api.MatchID]
            if last_id <= stop_id:
                index = min(i for i, m in enumerate(results)
                            if m[api.MatchID] <= stop_id)
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
            current_id = results[-1][api.MatchID]
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
        stop_id = self.storage.latest()
        if start_id is not None:
            # We're doing some backfilling, so don't mind the stop_id
            stop_id = None

        num_matches = 0
        data_gen = self.data(start_id=start_id, stop_id=stop_id)
        batched_data = self.batch(data_gen, batch_size=batch_size)
        for batch in batched_data:
            self.storage.store(batch)
            num_matches += len(batch)
            if num_matches >= limit:
                break
        return num_matches


def run(start_id: Optional[int], bucket_name: str, storage_path: str):
    """Run the data collection.

    Args:
        start_id: The match id from which to start collecting data.
        bucket_name: The bucket to save results to.
        storage_path: The path within the bucket to which to save the results.
    """
    api_call = api.Api().public_matches
    cache_filename = 'cache.json'
    bucket = google.cloud.storage.Client().bucket(bucket_name)
    strg = storage.Storage(bucket=bucket, storage_path=storage_path, cache_filename=cache_filename)
    collector = Collector(api_call=api_call, storage=strg)
    return collector.collect(
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
