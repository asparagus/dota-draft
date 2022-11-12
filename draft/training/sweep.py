"""Script to start a wandb sweep.

Example:
```
python -m draft.training.sweep start --data.path=data/training/20221102
python -m draft.training.sweep continue SWEEP_ID COUNT
```

Run `continue` on multiple terminals to run parallelly.
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
            'values': [128, 256, 512, 1024],
        },
        Arguments.MODEL_LEARNING_RATE.name: {
            'value': 1e-3,
        },
        Arguments.MODEL_WEIGHT_DECAY.name: {
            'value': 1e-5,
        },
        Arguments.MODEL_LAYERS.name: {
            'values': [
                [64, 64],
                [128, 32],
                [128, 64],
                [128, 128],
                [128, 64, 64],
                [64, 64, 64],
            ],
        }
    },
}


def start_sweep(args: argparse.Namespace):
    """Start a new sweep.

    Args:
        args: Parsed command-line arguments
    """
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    if args.name:
        SWEEP_CONFIGURATION['name'] = args.name
    SWEEP_CONFIGURATION['method'] = args.method
    SWEEP_CONFIGURATION['parameters'].update({
        'data.path': {'value': kwargs['data.path']}
    })
    print(yaml.dump(SWEEP_CONFIGURATION, indent=4))
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIGURATION, project=WANDB.project)
    if args.count:
        wandb.agent(sweep_id, function=train.main, count=args.count)


def continue_sweep(args: argparse.Namespace):
    """Continue runs for an existing sweep.

    Args:
        args: Parsed command-line arguments
    """
    wandb.agent(args.sweep_id, function=train.main, count=args.count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start or continue a sweep to optimize hyperparameters.')
    subparsers = parser.add_subparsers()
    start_parser = subparsers.add_parser('start', help='Start a new sweep')
    start_parser.add_argument('--count', default=0, type=int, help='Number of runs to start for the sweep')
    start_parser.add_argument('--name', default=None, help='Name for the sweep')
    start_parser.add_argument('--method', default='random', help='Method for the hyperparameter search', choices=['random', 'grid', 'bayes'])
    start_parser.add_argument('--data.path', required=True, help='Path to the data for training the model')
    start_parser.set_defaults(func=start_sweep)
    continue_parser = subparsers.add_parser('continue', help='Continue a sweep')
    continue_parser.add_argument('sweep_id', type=str, help='ID for the sweep to continue')
    continue_parser.add_argument('count', type=int, help='Number of runs to start for the sweep')
    continue_parser.set_defaults(func=continue_sweep)
    args = parser.parse_args()

    args.func(args)
