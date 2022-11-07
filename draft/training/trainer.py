"""Script to start training.

Example run:
```
python -m draft.training.trainer --data=data/training/20221102
```
"""
import argparse
import os
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
from draft.training.argument import Argument, Arguments, read_config
from draft.training.callbacks import LogOutputHistogram
from draft.training.ingestion import MatchDataset


ConfigArguments = [
    getattr(Arguments, arg)
    for arg in dir(Arguments)
    if not callable(getattr(Arguments, arg)) and not arg.startswith("__")
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--run_name', default=None, help='Name for the run')
    for arg in ConfigArguments:
        parser.add_argument(f'--{arg.name}', type=arg.type, required=arg.default is None, default=arg.default)
    args = parser.parse_args()

    logger = WandbLogger(
        project=WANDB.project,
        name=args.run_name,
    )
    varargs = {k: v for k, v in vars(args).items() if v is not None}
    print('Command line overrides:')
    print(yaml.dump(varargs))
    wandb.run.config.update(varargs)

    torch.manual_seed(read_config(Arguments.REPRODUCIBILITY_SEED))
    DATASET_CONFIG = {
        'bucket_name': GCS.bucket,
        'prefix': read_config(Arguments.DATA_PATH),
        'match_filter': (
            ValidMatchFilter() &
            HighRankMatchFilter(30)
        ),
    }
    DATALOADER_CONFIG = {
        'batch_size': read_config(Arguments.DATA_BATCH_SIZE),
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
    wrapper_config = ModelWrapperConfig(symmetric=read_config(Arguments.MODEL_SYMMETRIC))
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
        max_epochs=1000,
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
