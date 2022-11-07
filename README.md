# dota-draft

ML Application for drafting heroes in the Dota 2 game.

![A scheduled cloud function gets data from OpenDota and saves it to CloudStorage, a scheduled trainer trains the model on this data and a web application fetches the trained model for the users to consult it](diagram.png "Architecture diagram")

## Data Source
[OpenDota API](https://www.opendota.com/) is scraped to retrieve match ids along with the teams' drafts and outcome.

## Technologies used

- [Google Cloud Function](https://cloud.google.com/functions)
- [Google Cloud Pub/Sub](https://cloud.google.com/pubsub)
- [Google Cloud Scheduler](https://cloud.google.com/scheduler)
- [Google Cloud Dataflow](https://cloud.google.com/dataflow) / [Apache Beam](https://beam.apache.org/)
- [Google Cloud AI Platform](https://cloud.google.com/ai-platform/docs/technical-overview)

## Workflow Summary

1. Trigger data collection from CloudScheduler using a Pub/Sub topic
    - Cloud function collects matches from the OpenDota API and saves them to CloudStorage
2. Trigger manually a batch processing of data that filters matches of interest and saves them to CloudStorage
    - These files are set up for a training job and are split into train/val/test folders
3. Train a model to predict the outcome of a match given the draft
    -  Currently saving the model using Weights & Biases, working on training using Cloud AI Platform
4. The model is stored online and can be retrieved for evaluation

### Pending
5. Use the trained model to suggest heroes in a web UI using [torch-js](https://github.com/torch-js/torch-js)

### Configs
In order for the project to work as intended, you'll need to set up the following files:
- `draft/configs/gcs.yaml`
- `draft/configs/opendota.yaml`
- `draft/configs/wandb.yaml`

See the specific [README](draft/configs/README.md) for more details.
