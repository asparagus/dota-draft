import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping
from pytorch_lightning.loggers import WandbLogger
import torch.utils.data
import wandb

from draft.data.filter import HighRankMatchFilter, ValidMatchFilter
from draft.model.simplemodel import SimpleModel, SimpleModelConfig
from draft.training.ingestion import MatchDataset


if __name__ == '__main__':
    torch.manual_seed(1)
    PROJECT = 'dota-draft'
    DATASET_CONFIG = {
        'bucket_name': 'dota-draft',
        'prefix': 'data/training/20221021',
        'match_filter': (
            ValidMatchFilter() and
            HighRankMatchFilter(30)
        ),
    }
    DATALOADER_CONFIG = {
        'batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': False,
    }

    training_dataset = MatchDataset(
        **DATASET_CONFIG,
        blob_regex='train.*.json',
    )
    validation_dataset = MatchDataset(
        **DATASET_CONFIG,
        blob_regex='val.*.json',
    )
    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        **DATALOADER_CONFIG,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        **DATALOADER_CONFIG,
    )
    config = SimpleModelConfig(num_heroes=138, dimensions=(1,), symmetric=True)
    model = SimpleModel(config)
    experiment_name = 'baseline'
    wandb_logger = WandbLogger(name=experiment_name, project=PROJECT)
    trainer = pl.Trainer(
        # gpus=1,
        default_root_dir='saved',
        precision=16,
        limit_train_batches=10,
        max_epochs=20,
        check_val_every_n_epoch=2,
        callbacks=[
            early_stopping.EarlyStopping('val_loss'),
        ],
        logger=wandb_logger,
    )
    trainer.fit(
        model,
        train_loader,
        validation_loader,
    )
    checkpoint_path = 'saved/{experiment_name}-{epoch}.pth'.format(
        experiment_name=experiment_name,
        epoch=trainer.current_epoch,
    )
    trainer.save_checkpoint(checkpoint_path)
    wandb.save(checkpoint_path)
