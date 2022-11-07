"""Script to start training.

Example run:
```
python -m draft.training.trainer --bucket=dota-draft --data=data/training/20221102
```
"""
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.utils.data
import wandb

from draft.data.filter import HighRankMatchFilter, ValidMatchFilter
from draft.model.mlp import MLP, MLPConfig
from draft.model.wrapper import ModelWrapper, ModelWrapperConfig
from draft.training.ingestion import MatchDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--project', default='dota-draft', help='Wandb project')
    parser.add_argument('--bucket', default='dota-draft', help='GCS bucket to get the data from')
    parser.add_argument('--data', required=True, help='GCS directory to get the data from')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for data loader')
    parser.add_argument('--seed', default=1, type=int, help='Seed for determinism')
    parser.add_argument('--run_name', default=None, help='Name for the run')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    DATASET_CONFIG = {
        'bucket_name': args.bucket,
        'prefix': args.data,
        'match_filter': (
            ValidMatchFilter() &
            HighRankMatchFilter(30)
        ),
    }
    DATALOADER_CONFIG = {
        'batch_size': args.batch_size,
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
    wrapper_config = ModelWrapperConfig(symmetric=True)
    module = MLP(mlp_config)
    model = ModelWrapper(config=wrapper_config, module=module)

    logger = WandbLogger(project=args.project, name=args.run_name)
    checkpoint_path = os.path.join(wandb.run.dir, 'checkpoints')
    callbacks = [
        EarlyStopping('val_loss'),
        ModelCheckpoint(
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            dirpath=checkpoint_path,
            filename='chkpt-{epoch:02d}-{val_loss:.2f}',
        )
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
