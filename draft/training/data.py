from typing import Optional

import re
import tempfile

import numpy as np
import google.cloud.storage as gcs
import torch.utils.data

from draft.data.match import Match, Matches
from draft.data.filter import MatchFilter, HighRankMatchFilter, ValidMatchFilter


class MatchDataset(torch.utils.data.IterableDataset):
    """Class for an iterable dataset of dota matches."""

    def __init__(
            self,
            bucket_name: str,
            prefix: str,
            blob_regex: Optional[str] = None,
            match_filter: Optional[MatchFilter] = None,
        ):
        """Initialize the dataset from cloud storage.

        Args:
            bucket_name: The name of the GCS bucket to get the data from
            prefix: The prefix of files to get
            blob_regex: (Optional) Files that do not match this regex are discarded
            match_filter: (Optional) Matches that do not pass this filter are discarded
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.blob_regex = re.compile(blob_regex) if blob_regex else None
        self.match_filter = match_filter

    @classmethod
    def matches_from_file(cls, path: str) -> Matches:
        """Retrieve the matches stored in a given file.

        Args:
            path: The path to the file with matches in the local file system.
        """
        with open(path, 'r') as f:
            lines = f.readlines()
            return [Match.loads(l) for l in lines]

    @classmethod
    def draft_from_match(cls, match: Match) -> np.array:
        """Retrieve the vector of heroes drafted for this match.

        Args:
            match: The match to parse
        """
        return np.array(match.radiant_heroes + match.dire_heroes)

    @classmethod
    def numpy_from_match(cls, match: Match):
        """Retrieve the numpy features, label for this match.

        Args:
            match: The match to parse
        """
        draft = cls.draft_from_match(match)
        result = np.array([float(match.radiant_win)])
        return draft, result

    def __iter__(self):
        """Generates matches from the given cloud storage.

        Will filter files and matches according to the configured filters, perform parsing
        and return an iterator for tuples of data, label.
        """
        worker_info = torch.utils.data.get_worker_info()
        storage_client = gcs.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.prefix)

        if self.blob_regex is not None:
            blobs = filter(lambda blob: self.blob_regex.match(blob.name), blobs)

        if worker_info is not None:  # multi-process data loading, use every n-th blob
            blobs = (
                blob
                for i, blob in enumerate(blobs)
                if i % worker_info.num_workers == worker_info.id
            )

        for blob in blobs:
            temp = tempfile.NamedTemporaryFile(mode='w')
            blob.download_to_filename(temp.name)
            matches = self.matches_from_file(temp.name)
            if self.match_filter is not None:
                matches = filter(self.match_filter, matches)
            for match in matches:
                yield self.numpy_from_match(match)


# This code can be used to try this out.
if __name__ == '__main__':
    dataset = MatchDataset(
        bucket_name='dota-draft',
        prefix='data/training/20221021',
        blob_regex='.*.txt',
        match_filter=ValidMatchFilter() and HighRankMatchFilter(60),
    )
    batch_size = 128
    num_workers = 4
    pin_memory = True
    shuffle = False
    batched_dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    it = iter(batched_dataset)
    print(next(it))
