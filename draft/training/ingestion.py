"""Module that defines the classes to load and use training data."""
from typing import Optional

import glob
import os

import numpy as np
import torch.utils.data
import wandb

from draft.data.match import Match, Matches
from draft.data.filter import MatchFilter, HighRankMatchFilter, ValidMatchFilter


class MatchDataset(torch.utils.data.IterableDataset):
    """Class for an iterable dataset of dota matches."""

    def __init__(
            self,
            local_dir: str,
            glob: Optional[str] = '*.txt',
            match_filter: Optional[MatchFilter] = None,
        ):
        """Initialize the dataset from cloud storage.

        Args:
            local_dir: Local directory where this dataset is downloaded
            glob: (Optional) Files that do not match this regex are discarded
            match_filter: (Optional) Matches that do not pass this filter are discarded
        """
        self.local_dir = local_dir
        self.glob = glob
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
        result = match.radiant_win
        return draft, result

    def __iter__(self):
        """Generates matches from the given cloud storage.

        Will filter files and matches according to the configured filters, perform parsing
        and return an iterator for tuples of data, label.
        """
        worker_info = torch.utils.data.get_worker_info()
        files = glob.glob(os.path.join(self.local_dir, self.glob))

        if worker_info is not None:  # multi-process data loading, use every n-th file
            files = (
                file
                for i, file in enumerate(files)
                if i % worker_info.num_workers == worker_info.id
            )

        for file in files:
            matches = self.matches_from_file(file)
            if self.match_filter is not None:
                matches = filter(self.match_filter, matches)
            for match in matches:
                yield self.numpy_from_match(match)


# This code can be used to try this out.
if __name__ == '__main__':
    run = wandb.init(name='ingestion-test')
    artifact = run.use_artifact('asparagus/dota-draft/matches:v0', type='dataset')
    artifact_dir = artifact.download()
    dataset = MatchDataset(
        local_dir=artifact_dir,
        glob='train*',
        match_filter=ValidMatchFilter() & HighRankMatchFilter(60),
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
