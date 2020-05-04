"""Collect ids for training data.

This script scrapes the parsedMatches api to collect ids.

Usage:
    set DOTA_API_KEY=<KEY>
    set GOOGLE_APPLICATION_CREDENTIALS=<PATH>
    python -m dota_draft.collect --num_matches=<NUM> --bucket=<BUCKET> --file=data/ids/<FILENAME>
"""
import argparse
import logging

from dota_draft import api
from google.cloud import storage


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_matches',
        dest='num_matches',
        default=1000,
        type=int,
        help='Number of match IDs to collect.'
    )
    parser.add_argument(
        '--bucket',
        dest='bucket',
        required=True,
        help='Bucket to store the results.'
    )
    parser.add_argument(
        '--file',
        dest='file',
        required=True,
        help='File name to store the results.'
    )

    args = parser.parse_args(argv)

    storage_client = storage.Client()
    bucket = storage_client.bucket(args.bucket)
    blob = bucket.blob(args.file)

    api_client = api.Api()
    match_ids = []
    last_id = None
    while len(match_ids) < args.num_matches:
        parsed_matches = api_client.parsed_matches(last_id)
        retrieved_ids = [str(m['match_id']) for m in parsed_matches]
        logging.info('Retrieved %i ids' % len(retrieved_ids))
        match_ids.extend(retrieved_ids)
        last_id = match_ids[-1]

        if len(retrieved_ids) == 0:
            logging.warning('Api results ran out')
            break

    blob.upload_from_string('\n'.join(match_ids))
    logging.info('Uploaded %i ids to %s' % 
                 (len(match_ids), args.file))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
