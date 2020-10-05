"""Process match ids and retrieve the actual match data.

Performs API calls to retrieve the matches, strips unnecessary data and
filters out some matches.

Usage:
    set GOOGLE_APPLICATION_CREDENTIALS=<PATH>
    python -m draft.retrieve --input=<INPUT_FILE> --output=<OUTPUT_FILE>
"""
import argparse
import json
import logging

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from draft import api


LOBBY_TYPE_PRACTICE = 1
LOBBY_TYPE_RANKED = 7
LOBBY_TYPE_TOURNAMENT = 2

GAME_MODE_CAPTAINS_MODE = 2
GAME_MODE_RANKED_ALL_PICK = 22


class Retrieve(beam.DoFn):
    """Retrieve stuff."""

    def __init__(self, api_key):
        self.api = api.Api(api_key)

    def process(self, match_id):
        yield json.dumps(self.api.match(match_id))


class Strip(beam.DoFn):
    """Strip unnecessary data."""

    def process(self, match_data):
        try:
            match = json.loads(match_data)
            keep_cols = ('match_id', 'lobby_type', 'game_mode',
                         'duration', 'radiant_win')
            summary = {
                k: match.get(k)
                for k in keep_cols
            }

            picks_bans = match.get('picks_bans')
            if not picks_bans:
                logging.warning('Missing picks_bans')
                return

            summary['radiant_picks'] = [
                pb['hero_id']
                for pb in picks_bans
                if pb['is_pick'] and pb['team'] == 0]
            summary['dire_picks'] = [
                pb['hero_id']
                for pb in picks_bans
                if pb['is_pick'] and pb['team'] == 1]

            players = match.get('players')
            if not players:
                logging.warning('Missing players')
                return

            summary['player_rank_tiers'] = [
                player['rank_tier']
                for player in players
            ]

            yield json.dumps(summary)
        except Exception as e:
            logging.warning(e)


class Filter(beam.DoFn):
    """Drop some matches."""

    def filter_rank(self, player_rank_tiers):
        return all((rank_tier is None or int(rank_tier) >= 60)
                   for rank_tier in player_rank_tiers)

    def process(self, match_data):
        try:
            match = json.loads(match_data)
            if (match['lobby_type'] in [LOBBY_TYPE_PRACTICE,
                                        LOBBY_TYPE_RANKED,
                                        LOBBY_TYPE_TOURNAMENT] and
                match['game_mode'] in [GAME_MODE_CAPTAINS_MODE,
                                    GAME_MODE_RANKED_ALL_PICK] and
                match['duration'] > 600):
                if self.filter_rank(match['player_rank_tiers']):
                    yield match_data
        except Exception as e:
            logging.warning(e)


def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the retrieve pipeline."""

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
    parser.add_argument(
        '--dota_api_key',
        dest='dota_api_key',
        help='Api key to OpenDota.')
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_args.extend([
        '--runner=DataflowRunner',
        '--project=dota-drafter-291422',
        '--staging_location=gs://dota-drafter-291422/staging',
        '--temp_location=gs://dota-drafter-291422/tmp',
        '--job_name=retrieve-matches',
    ])

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as pipeline:
        (pipeline
            | 'Read match IDs' >> ReadFromText(known_args.input)
            | 'Retrieve matches' >> beam.ParDo(Retrieve(known_args.dota_api_key))
            | 'Strip data' >> beam.ParDo(Strip())
            | 'Filter data' >> beam.ParDo(Filter())
            | 'Write' >> WriteToText(known_args.output))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
