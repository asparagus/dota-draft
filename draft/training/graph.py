import argparse
import glob
import os
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import early_stopping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from draft.data import api
from draft import stats
from draft.training import data_ingestion


class HeroOneHotEncoding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = self.one_hot_encoding()
    
    def one_hot_encoding(self):
        embedding_weights = np.zeros((self.vocab_size, self.vocab_size))
        for i in range(1, self.vocab_size):
            embedding_weights[i, i] = 1

        embedding = nn.Embedding(
            self.vocab_size, self.vocab_size,
            padding_idx=0,
            _weight=torch.tensor(embedding_weights, dtype=torch.float))
        embedding.weight.requires_grad = False
        return embedding

    def forward(self, x):
        return self.embeddings(x)


class GraphEncoding(nn.Module):
    def __init__(self, embedding_size, neighborhood_embedding_size, depth):
        super().__init__()
        self.embedding_size = embedding_size
        self.depth = depth
        self.team_pooling_layers = nn.ModuleList([
            nn.Linear(embedding_size, neighborhood_embedding_size, bias=False)
            for i in range(depth)
        ])
        self.enemy_pooling_layers = nn.ModuleList([
            nn.Linear(embedding_size, neighborhood_embedding_size, bias=False)
            for i in range(depth)
        ])
        self.compacting_layers = nn.ModuleList([
            nn.Linear(embedding_size + 2 * neighborhood_embedding_size, embedding_size, bias=False)
            for i in range(depth)
        ])

    def forward(self, x):
        print('Forward')
        print(x.detach().cpu().numpy())
        h = F.normalize(x, dim=2)
        print(h.detach().cpu().numpy())
        for k in range(self.depth):
            print('Depth: %i' % k)
            radiant = h.narrow(1, 0, 5)
            dire = h.narrow(1, 5, 5)

            radiant_team, _ = torch.max(F.relu(self.team_pooling_layers[k](radiant)), dim=1, keepdim=True)
            dire_team, _ = torch.max(F.relu(self.team_pooling_layers[k](dire)), dim=1, keepdim=True)

            radiant_enemy, _ = torch.max(F.relu(self.enemy_pooling_layers[k](radiant)), dim=1, keepdim=True)
            dire_enemy, _ = torch.max(F.relu(self.enemy_pooling_layers[k](dire)), dim=1, keepdim=True)

            concat_radiant = torch.cat(
                [radiant,
                 radiant_team.expand(-1, 5, -1),
                 dire_enemy.expand(-1, 5, -1)], dim=2)
            concat_dire = torch.cat(
                [dire,
                 dire_team.expand(-1, 5, -1),
                 radiant_enemy.expand(-1, 5, -1)], dim=2)
            h = F.normalize(
                F.relu(self.compacting_layers[k](
                    torch.cat([concat_radiant, concat_dire], dim=1)
                )), dim=2
            )
            print(h.detach().cpu().numpy())

        return h


class Model(pl.LightningModule):

    def __init__(self, vocab_size, embedding_size, graph_depth):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embeddings = HeroOneHotEncoding(vocab_size)
        # self.linear = nn.Linear(vocab_size, embedding_size)
        self.graph = GraphEncoding(vocab_size, embedding_size, graph_depth)
        self.dropout = nn.Dropout(0.5)

        # self.hidden_layers = nn.ModuleList([
        #     nn.Linear(embedding_size * 2, 1024),
        #     nn.Linear(1024, 1024),
        # ])
        # self.batch_norms = nn.ModuleList([
        #     nn.BatchNorm1d(1024),
        #     nn.BatchNorm1d(1024),
        # ])

        self.final_layer = nn.Linear(vocab_size, 1)

    def forward(self, x):
        team = self.embeddings(x)#F.relu(self.linear(self.embeddings(x)))
        # radiant = team.narrow(1, 0, 5)  # .sum(dim=1, keepdim=False)
        # dire = team.narrow(1, 5, 5)  # .sum(dim=1, keepdim=False)

        graph_embedding = self.graph(team)
        radiant_graph_embedding = graph_embedding.narrow(1, 0, 5).sum(dim=1, keepdim=False)
        dire_graph_embedding = graph_embedding.narrow(1, 5, 5).sum(dim=1, keepdim=False)

        # radiant_combined = torch.cat([radiant_graph_embedding, dire_graph_embedding], dim=1)
        # dire_combined = torch.cat([dire_graph_embedding, radiant_graph_embedding], dim=1)

        # for layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
        #     radiant_combined = self.dropout(batch_norm(F.relu(layer(radiant_combined))))
        #     dire_combined = self.dropout(batch_norm(F.relu(layer(dire_combined))))

        radiant_final = self.final_layer(radiant_graph_embedding)
        dire_final = self.final_layer(dire_graph_embedding)

        logits = torch.cat([radiant_final, dire_final], dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs

    def training_step(self, batch, _):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def loss(self, y_hat, y):
        batch_size = y.shape[0]
        positive_labels = torch.zeros(batch_size).long().to(y_hat.device)
        negative_labels = torch.ones(batch_size).long().to(y_hat.device)
        positive_loss = torch.dot(y.narrow(1, 0, 1).view(-1), F.nll_loss(y_hat, positive_labels, reduction='none'))
        negative_loss = torch.dot(y.narrow(1, 1, 1).view(-1), F.nll_loss(y_hat, negative_labels, reduction='none'))
        return (positive_loss + negative_loss) / batch_size

    def validation_step(self, batch, _):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        predictions = (1 - torch.argmax(y_hat, dim=1))
        labels = (1 - torch.argmax(y, dim=1))
        accuracy = (predictions == labels).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', accuracy)
        return {'val_loss': loss, 'val_acc': accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=0.01)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to predict the winning team.')
    parser.add_argument('--embedding_size', default=256, type=int, required=True, help='Embedding size')
    parser.add_argument('--graph_depth', default=1, type=int, required=True, help='Graph NN depth')
    parser.add_argument('--train_data', required=True, help='Data glob for training')
    parser.add_argument('--val_data', required=True, help='Data glob for training')
    parser.add_argument('--stats_dir', required=False, help='Path to the stats directory')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--output', required=True, help='Path to save the output')
    parser.add_argument('--checkpoint', required=False, help='Path to saved checkpoint')
    args = parser.parse_args()
    torch.manual_seed(1)
    np.set_printoptions(threshold=sys.maxsize)

    ## Define & Load
    model = Model(130, args.embedding_size, args.graph_depth)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    ## Data
    train_files = glob.glob(args.train_data)
    val_files = glob.glob(args.val_data)
    num_train = len(train_files)
    # Exclude validation files from training files
    for i in range(len(train_files) - 1, -1, -1):
        for val_f in val_files:
            if os.path.samefile(train_files[i], val_f):
                train_files.pop(i)
    
    train_files = train_files[:1]

    smoothed_ingester = data_ingestion.MatchIngester(label_smoothing=0.1)
    normal_ingester = data_ingestion.MatchIngester()
    wr_ingester = data_ingestion.WinRateIngester(single_heroes=True, hero_pairs=False, hero_matchups=False)

    train_datasets = [smoothed_ingester.DatasetFromFile(f) for f in train_files]# + [wr_ingester.DatasetFromFile(args.stats_dir) for _ in range(30)]
    # train_datasets = [wr_ingester.DatasetFromFile(args.stats_dir)]
    val_datasets = [normal_ingester.DatasetFromFile(f) for f in val_files]

    training_loader = data_ingestion.DataLoaderFromDatasets(
        train_datasets, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, shuffle=True)
    val_loader = data_ingestion.DataLoaderFromDatasets(
        val_datasets, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)

    # Train
    trainer = pl.Trainer(gpus=1, precision=16, callbacks=[
        # early_stopping.EarlyStopping('val_loss'),
    ])
    trainer.fit(model, training_loader)#, val_loader)

    # Save
    torch.save(model.state_dict(), args.output)

    # # Display
    # for x, y in training_loader:
    #     y_hat = model(x)
    #     print(x)
    #     print(y, np.exp(y_hat.detach().numpy())) 

    ## Validate
    gpu = torch.cuda.device(0)
    model.eval()
    for dataset_name, dataset in (#('Win Rate', wr_loader),
                                  ('Validation', val_loader),):
        print(dataset_name)
        sum_acc = 0
        sum_loss = 0
        count = 0
        for x, y in dataset:
            weight =  y.size(0) / args.batch_size
            out = model.validation_step((x, y), None)
            sum_loss += out['val_loss'].item() * weight
            sum_acc += out['val_acc'].item() * weight
            count += weight

        print('-- Loss: %.2f' % (sum_loss / count))
        print('-- Accuracy: %.2f' % (sum_acc / count))

