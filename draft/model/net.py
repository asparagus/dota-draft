import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl


# define the LightningModule
class Net(pl.LightningModule):
    def __init__(self, num_heroes, dimensions):
        super().__init__()
        embedding_dim = dimensions[0]
        self.embeddings = nn.Embedding(num_heroes, embedding_dim)
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dimensions[-1], 1))
        self.sequential = nn.Sequential(*layers)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        hero_embeddings = self.embeddings(x)

        radiant_hero_embeddings = hero_embeddings.narrow(dim=1, start=0, length=5)
        dire_hero_embeddings = hero_embeddings.narrow(dim=1, start=5, length=5)
        radiant_embedding = radiant_hero_embeddings.sum(dim=1)
        dire_embedding = dire_hero_embeddings.sum(dim=1)
        draft_embedding = radiant_embedding - dire_embedding

        logits = self.sequential(draft_embedding)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        pred = self.activation(logits)
        accuracy = torch.sum(pred.round() == y.round()) / len(y)
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        return {'loss': loss, 'pred': pred, 'acc': accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
