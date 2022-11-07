"""Script to start training.

Example run:
```
python -m draft.training.trainer --data=data/training/20221102
```
"""
import argparse
import attrs
import os
from typing import Optional
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.utils.data
import wandb

from draft.data.filter import HighRankMatchFilter, ValidMatchFilter
from draft.model.mlp import MLP, MLPConfig
from draft.model.wrapper import ModelWrapper, ModelWrapperConfig
from draft.providers import GCS, WANDB
from draft.training.callbacks import LogOutputHistogram
from draft.training.ingestion import MatchDataset


@attrs.define
class DataConfig:
    """Configuration used for the data."""
    path: str
    batch_size: int


@attrs.define
class ModelConfig:
    """Configuration used for the model."""
    symmetric: bool


@attrs.define
class ReproducibilityConfig:
    """Configuration used for the reproducibility."""
    seed: int


@attrs.define
class TrainingConfig:
    """Configuration used for training."""
    data: DataConfig
    model: ModelConfig
    reproducibility: ReproducibilityConfig


def add_parser_arguments_recursively(parser: argparse.ArgumentParser, config_class: type, prefix: Optional[str] = ''):
    if attrs.has(config_class):
        fields = attrs.fields_dict(config_class)
        for name, attribute in fields.items():
            add_parser_arguments_recursively(parser, attribute.type, prefix=f'{prefix}.{name}' if prefix else name)
    elif prefix:
        parser.add_argument(f'--{prefix}', default=None, type=config_class)
    else:
        raise ValueError("Cannot create config for non-attrs class")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--config', default='draft/training/training_config.yaml', help='Config to use as a base')
    parser.add_argument('--data.path', required=True, help='GCS directory to get the data from')
    parser.add_argument('--run_name', default=None, help='Name for the run')
    args, config_override_args = parser.parse_known_args()

    config_parser = argparse.ArgumentParser(description='')
    add_parser_arguments_recursively(config_parser, TrainingConfig)
    overrides = config_parser.parse_args(config_override_args)
    override_dict = {'data.path': vars(args)['data.path']}
    for key, value in vars(overrides).items():
        if value is not None:
            override_dict[key] = value

    logger = WandbLogger(
        project=WANDB.project,
        name=args.run_name,
        config=args.config,
    )
    print('Command line overrides:')
    print(yaml.dump(override_dict))
    wandb.run.config.update(override_dict, allow_val_change=True)

    wandb_config = wandb.run.config
    torch.manual_seed(wandb_config['reproducibility.seed'])
    DATASET_CONFIG = {
        'bucket_name': GCS.bucket,
        'prefix': wandb_config['data.path'],
        'match_filter': (
            ValidMatchFilter() &
            HighRankMatchFilter(30)
        ),
    }
    DATALOADER_CONFIG = {
        'batch_size': wandb_config['data.batch_size'],
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': False,
    }

    training_dataset = MatchDataset(
        **DATASET_CONFIG,
        blob_regex='train.*.txt',
    )
    validation_dataset = MatchDataset(
        **DATASET_CONFIG,
        blob_regex='val.*.txt',
    )
    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        **DATALOADER_CONFIG,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        **DATALOADER_CONFIG,
    )
    mlp_config = MLPConfig(num_heroes=138, layers=[32, 16])
    wrapper_config = ModelWrapperConfig(symmetric=wandb_config['model.symmetric'])
    module = MLP(mlp_config)
    model = ModelWrapper(config=wrapper_config, module=module)

    checkpoint_path = os.path.join(wandb.run.dir, 'checkpoints')
    callbacks = [
        EarlyStopping('val_loss'),
        ModelCheckpoint(
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            dirpath=checkpoint_path,
            filename='chkpt-{epoch:02d}-{val_loss:.2f}',
        ),
        LogOutputHistogram(output_key='predictions', bins=10, range=(0, 1)),
    ]
    trainer = pl.Trainer(
        # gpus=1,
        # precision=16,
        default_root_dir='saved',
        max_epochs=5,
        check_val_every_n_epoch=2,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(
        model,
        train_loader,
        validation_loader,
    )
    wandb.save(os.path.join(checkpoint_path, '*.ckpt'), base_path=wandb.run.dir)
