"""Cloud funtions entry point.

Cloud functions defined in this file:
- collect
    Requests new match ids to the OpenDota API and saves them to cloud storage.

To deploy cloud functions run the following command from the root of the repository.

gcloud functions deploy collect \
    --entry-point=collect \
    --runtime=python39 \
    --trigger-http \
    --project=<PROJECT_NAME> \
    --source=. \
    --ingress-settings=all \
    --set-env-vars=<DOTA_API_KEY>

Check additional documentation at https://cloud.google.com/sdk/gcloud/reference/functions/deploy.

Setup file is no longer used as part of the setup of cloud functions, requirements must be listed
in requirements.txt.
"""
import functions_framework

from draft.data import collect as coll


@functions_framework.http
def collect(request):
    num_matches = coll.run(start_id=None, bucket_name='dota-draft', storage_path='data/matches')
    return 'Collected data from {} matches.'.format(num_matches)
