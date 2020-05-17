"""Data retrieval and processing functionalities."""
import logging

from google.cloud import storage

from src import api


def new_match_ids(most_recent_retrieved=None, max_matches=None, batch_size=1000):
    api_client = api.Api()

    buffer = []
    num_retrieved = 0

    last_retrieved_id = None
    while True:
        parsed_matches = api_client.parsed_matches(last_retrieved_id)
        retrieved_ids = [m['match_id'] for m in parsed_matches]

        # Collect only new matches
        if most_recent_retrieved is not None:
            retrieved_ids = [i for i in retrieved_ids if i > most_recent_retrieved]

        if not retrieved_ids:
            logging.warning('Api results ran out')
            break

        last_retrieved_id = retrieved_ids[-1]
        buffer.extend(retrieved_ids)

        # Yield a batch if the buffer has reached the right size
        if len(buffer) >= batch_size:
            yield buffer[:batch_size]
            buffer = buffer[batch_size:]

        num_retrieved += len(retrieved_ids)
        logging.info('Received %i new ids' % len(retrieved_ids))

    if buffer:
        yield buffer

    logging.info('Retrieved a total of %i new ids' % num_retrieved)
