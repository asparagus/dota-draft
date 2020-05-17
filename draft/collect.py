"""Collect ids for training data.

This script scrapes the parsedMatches api to collect ids.

Usage:
    set DOTA_API_KEY=<KEY>
    set GOOGLE_APPLICATION_CREDENTIALS=<PATH>
    python3 -m draft.collect --num_matches=<NUM> --bucket=<BUCKET> --file=data/ids/<FILENAME>
"""
import argparse
import datetime
import json
import logging
import math

from google.cloud import storage

from draft import api
from draft import data


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_matches',
        dest='num_matches',
        default=10000,
        type=int,
        help='Number of match IDs to collect.'
    )
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        default=1000,
        type=int,
        help='Number of match IDs to save to each file.'
    )
    parser.add_argument(
        '--bucket',
        dest='bucket',
        default='dota-draft',
        help='Bucket to store the results.'
    )
    parser.add_argument(
        '--file-pattern',
        dest='file',
        default=datetime.datetime.today().strftime(r'data/ids/%Y-%m-%d.csv'),
        help='File name to store the results. Defaults to data/ids/YYYY-mm-dd.csv'
    )
    parser.add_argument(
        '--cache',
        dest='cache',
        default='data/ids/cache.json',
        help='File to keep track of the last run.'
    )

    args = parser.parse_args(argv)
    if args.num_matches <= 0:
        return

    max_batches = math.ceil(args.num_matches / args.batch_size)
    max_num_digits = math.ceil(math.log10(max_batches))

    storage_client = storage.Client()
    bucket = storage_client.bucket(args.bucket)

    most_recent_retrieved = None
    try:
        cache_blob = bucket.blob(args.cache)
        cache_data = cache_blob.download_as_string()
        logging.info('Cache retrieved: %s' % cache_data.decode('utf8'))

        cache = json.loads(cache_data)
        most_recent_retrieved = cache['last_retrieved']
    except:
        logging.warning('Cache could not be retrieved')

    last_retrieved = most_recent_retrieved
    num_files = 0
    num_retrieved = 0
    for batch in data.new_match_ids(most_recent_retrieved=most_recent_retrieved,
                                    max_matches=args.num_matches,
                                    batch_size=args.batch_size):
        filename = args.file + '-' + str(num_files).zfill(max_num_digits)
        blob = bucket.blob(filename)

        blob.upload_from_string('\n'.join([str(i) for i in batch]))
        logging.info('Uploaded %i ids to %s' %  (len(batch), filename))

        num_files += 1
        num_retrieved += len(batch)
        last_retrieved = max(last_retrieved, batch[0])

    result = {
        'last_run': datetime.datetime.today().strftime(r'%Y-%m-%d'),
        'num_retrieved': num_retrieved,
        'last_retrieved': last_retrieved
    }

    cache_blob = bucket.blob(args.cache)
    cache_blob.upload_from_string(json.dumps(result))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
