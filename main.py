"""Cloud funtions entry point.

Cloud functions defined in this file:
- collect
    Requests new match ids to the OpenDota API and saves them to cloud storage.

To deploy cloud functions run the following command from the root of the repository.

gcloud functions deploy collect \
    --entry-point=collect \
    --runtime=python37 \
    --trigger-http \
    --project=dota-draft \
    --source=. \
    --ingress-settings=internal-only \
    --set-env-vars=<DOTA_API_KEY>

Check additional documentation at https://cloud.google.com/sdk/gcloud/reference/functions/deploy.
"""
from draft import collect as coll


def collect(request):
    coll.run()