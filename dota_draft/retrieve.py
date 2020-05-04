"""Process match ids and retrieve the actual match data.

Performs API calls to retrieve the matches, strips unnecessary data and 
filters out some matches.

Usage:
    set DOTA_API_KEY=<KEY>
    set GOOGLE_APPLICATION_CREDENTIALS=<PATH>
    python -m dota_draft.retrieve --input=<INPUT_FILE> --output=<OUTPUT_FILE>
"""
import argparse
import json
import logging

from dota_draft import api

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


LOBBY_TYPE_RANKED = 7
GAME_MODE_RANKED_ALL_PICK = 22


class Retrieve(beam.DoFn):
    """Retrieve stuff."""

    def __init__(self):
        self.api = api.Api()

    def process(self, match_id):
        return self.api.matches(match_id)


class Strip(beam.DoFn):
    """Strip unnecessary data."""

    def process(self, match):
        keep_cols = ('match_id', 'lobby_type', 'game_mode',
                     'duration', 'radiant_win')
        summary = {
            k: match[k]
            for k in keep_cols
        }

        summary['radiant_picks'] = [
            pick_ban['hero_id']
            for pick_ban in match['picks_bans']
            if pick_ban['is_pick'] and pick_ban['team'] == 0]
        summary['dire_picks'] = [
            pick_ban['hero_id']
            for pick_ban in match['picks_bans']
            if pick_ban['is_pick'] and pick_ban['team'] == 1]

        summary['player_rank_tiers'] = [
            player['rank_tier']
            for player in match['players']
        ]

        return summary


class Filter(beam.DoFn):
    """Drop some matches."""

    def filter_rank(self, player_rank_tiers):
        return all((rank_tier is None or int(rank_tier) >= 60)
                   for rank_tier in player_rank_tiers)

    def process(self, match):
        if (match['lobby_type'] == LOBBY_TYPE_RANKED and
            match['game_mode'] == GAME_MODE_RANKED_ALL_PICK and
            match['duration'] > 600):
            if self.filter_rank(match['player_rank_tiers']):
                yield json.dumps(match)


def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the wordcount pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        required=True,
        help='Input file to process.')
    parser.add_argument(
        '--output',
        dest='output',
        required=True,
        help='Output file to write results to.')
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_args.extend([
        '--runner=DirectRunner',
        '--project=dota-draft',
        '--staging_location=gs://dota-draft/staging',
        '--temp_location=gs://dota-draft/tmp',
        '--job_name=retrieve-matches',
    ])

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as pipeline:
        (pipeline
            | 'Read match IDs' >> ReadFromText(known_args.input)
            | 'Retrieve matches' >> beam.ParDo(Retrieve)
            | 'Strip data' >> beam.ParDo(Strip)
            | 'Filter data' >> beam.ParDo(Filter)
            | 'Write' >> WriteToText(known_args.output))

        result = pipeline.run()
        result.wait_until_finish()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()