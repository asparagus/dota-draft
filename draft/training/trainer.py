import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping
import torch.utils.data
from draft.model.net import Net
from draft.training.data import MatchDataset, ValidMatchFilter, HighRankMatchFilter


if __name__ == '__main__':
    torch.manual_seed(1)
    DATASET_CONFIG = {
        'bucket_name': 'dota-draft',
        'prefix': 'data/matches/68',
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
        blob_regex='.*/\\d+[123456789].json',
    )
    validation_dataset = MatchDataset(
        **DATASET_CONFIG,
        blob_regex='.*/\\d+0.json',
    )
    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        **DATALOADER_CONFIG,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        **DATALOADER_CONFIG,
    )
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
    )
    trainer.fit(
        Net(num_heroes=138, dimensions=[256, 256, 256]),
        train_loader,
        validation_loader,
    )
