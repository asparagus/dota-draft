"""Script to start a wandb sweep.

Example run:
```
python -m draft.training.sweep 20 --data.path=data/training/20221102
```
"""
import argparse
import yaml

import wandb

from draft.training import train
from draft.training.argument import Arguments
from draft.providers import WANDB


# Use this dictionary to set up parameters.
# See: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
SWEEP_CONFIGURATION = {
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss',
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
    },
    'parameters': {
        Arguments.DATA_BATCH_SIZE.name: {
            'values': [64, 128, 256, 512],
        },
        Arguments.MODEL_LEARNING_RATE.name: {
            'values': [1e-3, 1e-4, 1e-5],
        },
        Arguments.MODEL_WEIGHT_DECAY.name: {
            'values': [1e-2, 1e-3, 1e-4, 1e-5],
        },
        Arguments.MODEL_LAYERS.name: {
            'values': [
                [16, 16],
                [32, 16],
                [64, 32],
                [64, 64],
                [128, 32],
                [128, 64],
            ],
        }
    },
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a sweep to optimize hyperparameters.')
    parser.add_argument('count', type=int, help='Number of runs to start for the sweep')
    parser.add_argument('--name', default=None, help='Name for the sweep')
    parser.add_argument('--method', default='random', help='Method for the hyperparameter search', choices=['random', 'grid', 'bayes'])
    parser.add_argument('--data.path', required=True, help='Path to the data for training the model')
    args = parser.parse_args()

    SWEEP_CONFIGURATION.update({
        'name': args.name,
        'method': args.method,
    })

    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    SWEEP_CONFIGURATION['parameters'].update({
        'data.path': {'value': kwargs['data.path']}
    })
    print(yaml.dump(SWEEP_CONFIGURATION, indent=4))
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIGURATION, project=WANDB.project)
    wandb.agent(sweep_id, function=train.main, count=args.count)
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
