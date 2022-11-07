import numpy as np
import wandb
from pytorch_lightning.callbacks import Callback


class LogOutputHistogram(Callback):
    def __init__(self, output_key, **kwargs):
        self.output_key = output_key
        self.kwargs = kwargs

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        hist = np.histogram(outputs[self.output_key], **self.kwargs)
        wandb.log({self.output_key: wandb.Histogram(np_histogram=hist)})
