import abc
import json
import re
import tempfile
from typing import Dict, List, Optional

import google.cloud.storage as gcs
import numpy as np
import torch.utils.data


class MatchFilter(abc.ABC):
    """Base class to implement a filter for stored matches.
    
    Filters must implement a __call__ function that returns a bool
    indicating whether the match should be kept.
    """

    @abc.abstractmethod
    def __call__(self, match: Dict):
        """Runs on the match and determines whether it should be kept.
        
        Args:
            match: The match to evaluate
        """
        raise NotImplementedError()

    def __or__(self, other: 'MatchFilter'):
        """Defines the or operator between this and another filter."""
        return Conjunction(self, other)

    def __and__(self, other: 'MatchFilter'):
        """Defines the and operator between this and another filter."""
        return Conjunction(self, other)

    def __not__(self):
        """Defines the not operator for this filter."""
        return Negation(self)


class Disjunction(MatchFilter):
    """A disjunction between two filters, will keep the match when any is true."""

    def __init__(self, filter_a, filter_b):
        """Initialize the disjunction with both filters.

        Args:
            filter_a: One of the filters
            filter_b: The other filter
        """
        self.filter_a = filter_a
        self.filter_b = filter_b

    def __call__(self, match):
        """Evaluates both filters and returns the or between them.

        Args:
            match: The match to evaluate
        """
        return self.filter_a(match) or self.filter_b(match)


class Conjunction(MatchFilter):
    """A conjunction between two filters, will keep the match when both are true."""

    def __init__(self, filter_a, filter_b):
        """Initialize the conjunction with both filters.

        Args:
            filter_a: One of the filters
            filter_b: The other filter
        """
        self.filter_a = filter_a
        self.filter_b = filter_b

    def __call__(self, match):
        """Evaluates both filters and returns the and between them.

        Args:
            match: The match to evaluate
        """
        return self.filter_a(match) and self.filter_b(match)


class Negation(MatchFilter):
    """A negation of a filter, will keep the match when the filter is false."""

    def __init__(self, filter):
        """Initialize the negation of the filter.

        Args:
            filter: The filter to negate
        """
        self.filter = filter

    def __call__(self, match):
        """Evaluates the filter and returns the opposite result.

        Args:
            match: The match to evaluate
        """
        return not self.filter(match)


class ValidMatchFilter(MatchFilter):
    """A filter for removing invalid matches where the draft did not finish."""

    def __call__(self, match):
        """Evaluate whether the match had 5 heroes on both sides.

        Args:
            match: The match to evaluate
        """
        return (
            len(match['radiant_team'].split(',')) == 5 and
            len(match['dire_team'].split(',')) == 5
        )


class HighRankMatchFilter(MatchFilter):
    """A filter for removing matches below a certain rank."""

    def __init__(self, minimum_rank: int):
        """Initialize the filter with a minimum rank value.

        Args:
            minimum_rank: The minimum acceptable rank, matches
                below this will be discarded"""
        self.minimum_rank = minimum_rank

    def __call__(self, match):
        """Evaluate whether the match has the required rank.

        Args:
            match: The match to evaluate
        """
        return (
            match['avg_rank_tier'] >= self.minimum_rank
        )


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
    def matches_from_file(cls, path: str) -> List[Dict]:
        """Retrieve the matches stored in a given file.

        Args:
            path: The path to the file with matches in the local file system.
        """
        with open(path, 'r') as f:
            return json.load(f)

    @classmethod
    def draft_from_match(cls, match: Dict) -> np.array:
        """Retrieve the vector of heroes drafted for this match.

        Args:
            match: The match to parse
        """
        all_heroes = (
            match['radiant_team'].split(',') +
            match['dire_team'].split(',')
        )
        return np.array([int(h) for h in all_heroes])

    @classmethod
    def numpy_from_match(cls, match: Dict):
        """Retrieve the numpy features, label for this match.

        Args:
            match: The match to parse
        """
        draft = cls.draft_from_match(match)
        result = int(match['radiant_win'])
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
        prefix='data/matches',
        blob_regex='.*/\\d+.json',
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
