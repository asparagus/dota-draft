import argparse
import json
import logging
import requests

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


PROJECT = 'dota-draft'
MATCHES_URL = 'https://api.opendota.com/api/matches/%s'


def Retrieve(match_id):
  url = MATCHES_URL % match_id
  response = requests.get(url)
  if response.ok:
    yield response.text


def Summarize(match_data):
  match = json.loads(match_data)
  match_summary = {
    k: match[k]
    for k in ('match_id',
              'lobby_type',
              'game_mode',
              'duration',
              'radiant_win')
  }

  match_summary['radiant_picks'] = [
    pick_ban['hero_id']
    for pick_ban in match['picks_bans']
    if pick_ban['is_pick'] and pick_ban['team'] == 0]
  match_summary['dire_picks'] = [
    pick_ban['hero_id']
    for pick_ban in match['picks_bans']
    if pick_ban['is_pick'] and pick_ban['team'] == 1]

  match_summary['player_rank_tiers'] = [
    player['rank_tier']
    for player in match['players']
  ]

  return json.dumps(match_summary)



def run(argv=None, save_main_session=True):
  """Main entry point; defines and runs the wordcount pipeline."""

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      dest='input',
      default='gs://dataflow-samples/shakespeare/kinglear.txt',
      help='Input file to process.')
  parser.add_argument(
      '--output',
      dest='output',
      default='gs://dota-draft/matches/',
      help='Output file to write results to.')
  known_args, pipeline_args = parser.parse_known_args(argv)
  pipeline_args.extend([
      '--runner=DataflowRunner',
      '--project=dota-draft',
      '--staging_location=gs://dota-draft/staging',
      '--temp_location=gs://dota-draft/tmp',
      '--job_name=request-matches',
  ])

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  with beam.Pipeline(options=pipeline_options) as p:

    # Read the text file[pattern] into a PCollection.
    match_ids = p | ReadFromText(known_args.input)

    # Count the occurrences of each word.

    matches = match_ids | beam.FlatMap(Retrieve)
    summaries = matches | beam.Map(Summarize)

    # Write the output using a "Write" transform that has side effects.
    # pylint: disable=expression-not-assigned
    summaries | WriteToText(known_args.output)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()