"""Apache Beam pipeline for compacting a dataset for training.

python -m draft.data.compact \
    --runner DataflowRunner \
    --region us-west2 \
    --input gs://dota-draft/data/matches/68*.json \
    --output gs://dota-draft/data/training/20221021 \
    --project dota-draft \
    --temp_location gs://dota-draft/tmp \
    --setup_file ./setup.py

"""
import argparse
import json
import logging
import os

import numpy as np
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from draft.data.match import Match
from draft.data.filter import HighRankMatchFilter, ValidMatchFilter


class MatchSplitter(beam.DoFn):
    """Class that splits the saved JSONs into multiple lines with one match each."""
    def process(self, collection: str):
        """Process a collection of matches and splits them into single match-strings.

        Args:
            collection: A string representing a JSON list with multiple matches
        """
        matches = json.loads(collection)
        for m in matches:
            yield json.dumps(m)


class Split:
    """Class defining a train/val/test split."""

    def __init__(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """Initialize the instance with given ratios.

        Ratios must add up to 1.0

        Args:
            train_ratio: Ratio (0.0-1.0) for the training data
            val_ratio: Ratio (0.0-1.0) for the validation data
            test_ratio: Ratio (0.0-1.0) for the test data
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.dist = np.cumsum([train_ratio, val_ratio])
        assert (train_ratio + val_ratio + test_ratio == 1.0)

    def partition(self, match: Match, num_partitions: int):
        """Assign a partition to a given match based on the hash of its id.

        Args:
            match: The match to assign a partition to
            num_partitions: The number of partitions used, must be 3.
        """
        assert num_partitions == 3
        h = hash(match.match_id)
        f = float(h % 1000) / 1000.0
        return np.searchsorted(self.dist, f)

    @property
    def partition_names(self):
        """Get the names of the partitions, to be used for the storage path."""
        return ['train', 'val', 'test']


def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the wordcount pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        required=True,
        help='Cloud storage glob of files to process.')
    parser.add_argument(
        '--output',
        dest='output',
        required=True,
        help='Output location for the processed files.')
    parser.add_argument(
        '--minimum_rank',
        dest='minimum_rank',
        type=int,
        required=False,
        default=60,
        help='Minimum rank of matches to collect.')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--no-split', dest='split', action='store_false')
    parser.set_defaults(split=True)

    known_args, pipeline_args = parser.parse_known_args(argv)

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    # The pipeline will be run on exiting the with block.
    with beam.Pipeline(options=pipeline_options) as p:

        matches = (
            p
            | 'Read' >> ReadFromText(known_args.input)
            | 'Splitter' >> beam.ParDo(MatchSplitter())
            | 'Parse' >> beam.Map(Match.loads)
            | 'Filter valid' >> beam.Filter(ValidMatchFilter())
            | 'Filter high rank' >> beam.Filter(HighRankMatchFilter(minimum_rank=60))
            | 'Shuffle' >> beam.Reshuffle()
        )

        partitions_and_outputs = []
        if known_args.split:
            splitter = Split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
            partition_names = splitter.partition_names
            partitions = matches | 'Partition' >> beam.Partition(splitter.partition, len(partition_names))
            partition_output_paths = [
                os.path.join(known_args.output, partition_name)
                for partition_name in partition_names
            ]
            partitions_and_outputs = zip(partitions, partition_names, partition_output_paths)
        else:
            partitions_and_outputs = [(matches, '', known_args.output)]

        for partition, name, output in partitions_and_outputs:
            (partition
                | 'Serialize {}'.format(name) >> beam.Map(Match.dumps)
                | 'Write {}'.format(name) >> WriteToText(output, file_name_suffix='.txt'))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
