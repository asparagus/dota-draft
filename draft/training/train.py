"""Script to start training.

Example run:
```
python -m draft.training.train --data.artifact_id=asparagus/dota-draft/matches:v0
```
"""
from typing import List, Optional
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.utils.data
import wandb

from draft.data.filter import HighRankMatchFilter, ValidMatchFilter
from draft.model.embedding import EmbeddingConfig
from draft.model.match_prediction import MatchPredictionConfig
from draft.model.modules.mlp import MlpConfig
from draft.model.model import Model, ModelConfig
from draft.model.team_modules import TeamConvolutionConfig
from draft.providers import WANDB
from draft.training.argument import Arguments, read_config
from draft.training.callbacks import OutputLoggerCallback, WeightLoggerCallback
from draft.training.ingestion import MatchDataset


ConfigArguments = [
    getattr(Arguments, arg)
    for arg in dir(Arguments)
    if not callable(getattr(Arguments, arg)) and not arg.startswith("__")
]


def create_logger(run_name: Optional[str] = None):
    """Create a logger (and initialize the run).

    Args:
        run_name: (Optional) Recognizable name to use for the run
    """
    return WandbLogger(
        project=WANDB.project,
        name=run_name,
    )


def export(
        checkpoint_dir: str,
        checkpoint: str,
    ) -> wandb.Artifact:
    """Export a model checkpoint and return the wandb artifact.

    The artifact contains both the exported model and original checkpoint.

    Args:
        checkpoint_dir: The directory where the checkpoints are stored
        checkpoint: The name of the checkpoint to load before exporting
    """
    ckpt_path = os.path.join(checkpoint_dir, checkpoint)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(
        local_path=ckpt_path,
        name='model.ckpt',
    )

    onnx_path = ckpt_path.replace('.ckpt', '.onnx')
    model = Model.load_from_checkpoint(ckpt_path)
    model.export(onnx_path)

    artifact.add_file(
        local_path=onnx_path,
        name='model.onnx',
    )
    return artifact


def train(logger: WandbLogger):
    """Run training with an initialized logger."""
    torch.manual_seed(read_config(Arguments.REPRODUCIBILITY_SEED))
    artifact = wandb.run.use_artifact(
        read_config(Arguments.DATA_ARTIFACT_ID),
        type='dataset',
    )
    artifact_dir = artifact.download()
    DATASET_CONFIG = {
        'local_dir': artifact_dir,
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
        glob='train*',
    )
    validation_dataset = MatchDataset(
        **DATASET_CONFIG,
        glob='val*',
    )
    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        **DATALOADER_CONFIG,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        **DATALOADER_CONFIG,
    )
    model_config = ModelConfig(
        embedding_config=EmbeddingConfig(
            num_heroes=read_config(Arguments.MODEL_NUM_HEROES),
            embedding_size=read_config(Arguments.MODEL_EMBEDDING_SIZE),
        ),
        team_convolution_config=TeamConvolutionConfig(
            input_dimension=read_config(Arguments.MODEL_EMBEDDING_SIZE),
            layers=read_config(Arguments.MODEL_TEAM_CONVOLUTION_LAYERS),
            activation=True,
        ),
        match_prediction_config=MatchPredictionConfig(
            symmetric=read_config(Arguments.MODEL_SYMMETRIC),
            mlp_config=MlpConfig(
                input_dimension=read_config(Arguments.MODEL_TEAM_CONVOLUTION_LAYERS)[-1],
                layers=read_config(Arguments.MODEL_LAYERS),
            ),
        ),
        learning_rate=read_config(Arguments.MODEL_LEARNING_RATE),
        weight_decay=read_config(Arguments.MODEL_WEIGHT_DECAY),
    )
    model = Model(config=model_config)

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
        # OutputLoggerCallback(
        #     output_key='predictions',
        #     on_validation_batch_end=True,
        #     bins=10,
        #     range=(0, 1),
        # ),
        # WeightLoggerCallback(
        #     on_train_end=True,
        # ),
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

    # Export and generate trained model artifacts
    checkpoints = os.listdir(checkpoint_path)
    for ckpt in checkpoints:
        wandb.run.log_artifact(
            export(checkpoint_path, ckpt)
        )


def main(**kwargs):
    """Run training, use kwargs to update wandb config.

    Args:
        **kwargs: Keyword arguments for the wandb config
    """
    logger = create_logger()
    wandb.run.config.update(kwargs)
    train(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--run_name', default=None, help='Name for the run')
    for arg in ConfigArguments:
        _type = arg.type
        if _type == List[int]:
            parser.add_argument(f'--{arg.name}', type=int, required=arg.default is None, default=arg.default, nargs='*')
        elif _type == List[float]:
            parser.add_argument(f'--{arg.name}', type=float, required=arg.default is None, default=arg.default, nargs='*')
        elif _type == List[str]:
            parser.add_argument(f'--{arg.name}', type=str, required=arg.default is None, default=arg.default, nargs='*')
        elif _type in (str, int, float):
            parser.add_argument(f'--{arg.name}', type=_type, required=arg.default is None, default=arg.default)
        elif _type == bool:
            parser.add_argument(f'--{arg.name}', type=_type, required=arg.default is None, default=arg.default, action=argparse.BooleanOptionalAction)
        else:
            raise NotImplementedError('Argument type not supported: {}'.format(_type))
    args = parser.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    main(**kwargs)
