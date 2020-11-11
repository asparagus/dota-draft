import argparse
import glob
import json
import os
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import early_stopping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data


def DatasetFromFile(file):
    def parse_team(team_str):
        return [int(i) for i in team_str.split(',')]

    def parse_line(line):
        match = json.loads(line)
        return (parse_team(match['radiant_team'])
                if match['radiant_win']
                else parse_team(match['dire_team']))

    with open(file, 'r') as f:
        matches = [parse_line(line)
                   for line in f.read().split('\n')
                   if line]
        matches = torch.tensor(
            [heroes for heroes in matches if len(heroes) == 5],
            dtype=torch.long)
        return data.TensorDataset(matches)


class CBOW(pl.LightningModule):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        embeds = self.embeddings(x)
        embed_sum = embeds.sum(1, keepdim=True)
        split_embeds = [
            embed_sum - embeds.narrow(1, i, 1)
            for i in range(5)
        ]
        outs = [
            F.relu(self.linear1(embed))
            for embed in split_embeds
        ]
        outs = [
            self.linear2(out)
            for out in outs
        ]
        out = torch.cat(outs, 1)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def training_step(self, batch, _):
        # training_step defined the train loop. It is independent of forward
        x, = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = F.nll_loss(x_hat.view(-1, self.vocab_size), x.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        # training_step defined the train loop. It is independent of forward
        x, = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = F.nll_loss(x_hat.view(-1, self.vocab_size), x.view(-1))
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BOW embeddings on winning team drafts.')
    parser.add_argument('--train_data', required=True, help='Data glob for training')
    parser.add_argument('--val_data', required=True, help='Data glob for training')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
    parser.add_argument('--embedding_dim', required=True, type=int, help='Learned embedding dimension')
    parser.add_argument('--output', required=True, help='Path to save the output')
    args = parser.parse_args()
    torch.manual_seed(1)

    train_files = glob.glob(args.train_data)
    val_files = glob.glob(args.val_data)
    num_train = len(train_files)
    # Exclude validation files from training files
    for i in range(len(train_files) - 1, -1, -1):
        for val_f in val_files:
            if os.path.samefile(train_files[i], val_f):
                train_files.pop(i)

    training_dataset = data.ConcatDataset([
        DatasetFromFile(f)
        for f in train_files
    ])
    val_dataset = data.ConcatDataset([
        DatasetFromFile(f)
        for f in val_files
    ])
    training_loader = data.DataLoader(
        training_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4)

    model = CBOW(130, args.embedding_dim)

    trainer = pl.Trainer(gpus=1, precision=16, callbacks=[
        early_stopping.EarlyStopping('val_loss'),
    ])
    trainer.fit(model, training_loader, val_loader)

    torch.save(
        {'embeddings.weight': model.state_dict()['embeddings.weight']},
        args.output
    )

    # Test
    model.eval()
    draft = [[
        52,  # Leshrac
        55,  # Dark Seer
        9,   # Mirana
        26,  # Lion
        1,   # Anti-Mage
    ]]

    draft_tensor = torch.tensor(draft, dtype=torch.long)
    output = model(draft_tensor)
    prediction = torch.argmax(output, dim=2)
    print(prediction)
