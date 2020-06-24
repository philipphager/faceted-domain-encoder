import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils import data

from experiments.classification.util.optimizer import RAdam


class Classifier(LightningModule):
    def __init__(self,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 input_dims: int,
                 hidden_dims: int,
                 output_dims: int):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.input = nn.Linear(input_dims, hidden_dims)
        self.hidden = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)
        self.norm = nn.LayerNorm(hidden_dims)
        self.criterion = nn.BCELoss()
        self.dropout = nn.Dropout(0.3)

    def prepare_data(self):
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train)

    def forward(self, x):
        x = self.input(x)
        x = self.norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = F.sigmoid(x)
        return x

    def training_step(self, batch, batch_index):
        x, y = batch
        y_predict = self.forward(x)
        loss = self.criterion(y_predict, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=0.003)

    def train_dataloader(self):
        dataset = Dataset(self.x_train, self.y_train)
        loader = data.DataLoader(dataset, batch_size=32, num_workers=4)
        return loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_predict = self.forward(x)
        loss = self.criterion(y_predict, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def val_dataloader(self):
        dataset = Dataset(self.x_val, self.y_val)
        loader = data.DataLoader(dataset, batch_size=32, num_workers=4)
        return loader

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_predict = self.forward(x)
        loss = self.criterion(y_predict, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': avg_loss}

    def test_dataloader(self):
        dataset = Dataset(self.x_test, self.y_test)
        loader = data.DataLoader(dataset, batch_size=32, num_workers=4)
        return loader


class Dataset(data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
