"""Cloud funtions entry point.

Cloud functions defined in this file:
- collect
    Requests new match ids to the OpenDota API and saves them to cloud storage.

To deploy cloud functions run the following command from the root of the repository.
Use a Pub/Sub topic for triggering to avoid triggering multiple times.

gcloud functions deploy collect \
    --entry-point=collect \
    --runtime=python39 \
    --trigger-topic=<TOPIC_NAME> \
    --project=<PROJECT_NAME> \
    --source=. \
    --ingress-settings=all \
    --ignore-file=.gitignore \
    --build-env-vars-file=environments/collect/env.yaml

Check additional documentation at https://cloud.google.com/sdk/gcloud/reference/functions/deploy.
"""
import functions_framework

from draft.data import collect as coll
from draft.providers import GCS


@functions_framework.http
def collect(request):
    num_matches = coll.run(start_id=None, bucket_name=GCS.bucket, storage_path='data/matches')
    return 'Collected data from {} matches.'.format(num_matches)
