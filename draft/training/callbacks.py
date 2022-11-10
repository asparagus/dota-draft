"""Module for defining special Callbacks to use during training."""
import numpy as np
import wandb
from pytorch_lightning.callbacks import Callback


class WeightLoggerCallback(Callback):
    """Class that logs weights."""

    def __init__(
        self,
        on_train_start: bool = False,
        on_train_end: bool = False,
        on_save_checkpoint: bool = False,
        on_train_batch_end: bool = False,
    ):
        """Initialize the WeightLoggercallback.

        Args:
            on_train_start: Whether to run this when the training starts
            on_train_end: Whether to run this when the training ends
            on_save_checkpoint: Whether to run this when saving a checkpoint
            on_train_batch_end: Whether to run this when a training batch ends
        """
        self._on_train_start = on_train_start
        self._on_train_end = on_train_end
        self._on_save_checkpoint = on_save_checkpoint
        self._on_train_batch_end = on_train_batch_end

    def weights(self, pl_module):
        """Create and return a wandb table summarizing the weights.

        Args:
            pl_module: The module from which to get the weights
        """
        embeddings = pl_module.module.embeddings.weight.detach().numpy()
        table = wandb.Table(columns=['embedding'])
        for e in embeddings:
            table.add_data(e)
        return table

    def on_train_start(self, trainer, pl_module):
        if self._on_train_start:
            wandb.log({'embeddings': self.weights(pl_module)})

    def on_train_end(self, trainer, pl_module):
        if self._on_train_end:
            wandb.log({'embeddings': self.weights(pl_module)})

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self._on_save_checkpoint:
            wandb.log({'embeddings': self.weights(pl_module)})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._on_train_batch_end:
            wandb.log({'embeddings': self.weights(pl_module)})


class OutputLoggerCallback(Callback):
    """Class that logs outptus."""

    def __init__(
        self,
        output_key: str,
        on_train_batch_end: bool = False,
        on_predict_epoch_end: bool = False,
        on_validation_batch_end: bool = False,
        on_test_batch_end: bool = False,
        on_predict_batch_end: bool = False,
        **histogram_kwargs,
    ):
        """Initialize the OutputLoggerCallback.

        Args:
            output_key: The key to retrieve and log from the output
            on_train_batch_end: Whether to run this when a training batch ends
            on_predict_epoch_end: Whether to run this when a prediction epoch ends
            on_validation_batch_end: Whether to run this when a validation batch ends
            on_test_batch_end: Whether to run this when a test batch ends
            on_predict_batch_end: Whether to run this when a prediction batch ends
            **histogram_kwargs: Additional kwargs to pass on to np.histogram
        """
        self.output_key = output_key
        self._on_train_batch_end = on_train_batch_end
        self._on_predict_epoch_end = on_predict_epoch_end
        self._on_validation_batch_end = on_validation_batch_end
        self._on_test_batch_end = on_test_batch_end
        self._on_predict_batch_end = on_predict_batch_end
        self.histogram_kwargs = histogram_kwargs

    def histogram(self, value):
        """Create and return a wandb.Histogram from the given data.

        Args:
            value: Value retrieved from the model outputs
        """
        return wandb.Histogram(np_histogram=np.histogram(value, **self.histogram_kwargs))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._on_train_batch_end:
            wandb.log({self.output_key: self.histogram(outputs[self.output_key])})

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        if self._on_predict_epoch_end:
            wandb.log({self.output_key: self.histogram(outputs[self.output_key])})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._on_validation_batch_end:
            wandb.log({self.output_key: self.histogram(outputs[self.output_key])})

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._on_test_batch_end:
            wandb.log({self.output_key: self.histogram(outputs[self.output_key])})

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._on_predict_batch_end:
            wandb.log({self.output_key: self.histogram(outputs[self.output_key])})
