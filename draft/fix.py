"""Collect ids for training data.

This script scrapes the parsedMatches api to collect ids.

Usage:
    set DOTA_API_KEY=<KEY>
    set GOOGLE_APPLICATION_CREDENTIALS=<PATH>
    python3 -m src.collect --num_matches=<NUM> --bucket=<BUCKET> --file=data/ids/<FILENAME>
"""
import json
from google.cloud import storage


def run(argv=None):
    result = {
        'earliest': 6454017406,
        'latest': 6466296506
    }

    storage_client = storage.Client()
    bucket = storage_client.bucket('dota-draft')
    cache_blob = bucket.blob('data/matches/cache.json')
    cache_blob.upload_from_string(json.dumps(result))


if __name__ == '__main__':
    run()
