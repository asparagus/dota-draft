import argparse
import json
import os

from . import model

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help = 'GCS path to data. We assume that data is in gs://BUCKET/babyweight/preproc/',
        required = True
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--batch_size',
        help = 'Number of examples to compute gradient over.',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )
    parser.add_argument(
        '--num_heroes',
        help = 'Number of dota hero ids. Do not modify unless new heroes are added.',
        type = int,
        default = 130
    )

    parser.add_argument(
        '--train_examples',
        help = 'Number of examples (in thousands) to run the training job over. If this is more than actual # of examples available, it cycles through them. So specifying 1000 here when you have only 100k examples makes this 10 epochs.',
        type = int,
        default = 5000
    )
    parser.add_argument(
        '--train_pattern',
        help = 'Specify a pattern for the training files.',
        default = 'data/records/train/*.tfrecords'
    )
    parser.add_argument(
        '--eval_pattern',
        help = 'Specify a pattern for the evaluation files.',
        default = 'data/records/val/*.tfrecords'
    )
    parser.add_argument(
        '--eval_interval',
        help = 'Seconds in between evaluating the model. Default to 300 (every 5 minutes).',
        type = int,       
        default = 300
    )
    parser.add_argument(
        '--learning_rate',
        help = 'Learning rate',
        type = float,
        default = 0.001
    )
        
    ## parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    ## assign the arguments to the model variables
    output_dir = arguments.pop('output_dir')
    model.BUCKET = arguments.pop('bucket')
    model.BATCH_SIZE = arguments.pop('batch_size')
    model.TRAIN_EXAMPLES = arguments.pop('train_examples') * 1000
    model.EVAL_INTERVAL = arguments.pop('eval_interval')
    model.NUM_HEROES = arguments.pop('num_heroes')
    model.TRAIN_PATTERN = arguments.pop('train_pattern')
    model.EVAL_PATTERN = arguments.pop('eval_pattern')

    model.LEARNING_RATE = arguments.pop('learning_rate')
    print ("Will train on {} examples using batch_size={}".format(model.TRAIN_EXAMPLES, model.BATCH_SIZE))

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # Run the training job
    model.train_and_evaluate(output_dir)