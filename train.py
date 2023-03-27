import hydra
from omegaconf import DictConfig
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from collections import OrderedDict
import lightning.pytorch as pl

from lightning.pytorch.loggers import TensorBoardLogger


# Download training and testing data
class FMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '..', batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    
    def prepare_data(self):
        FashionMNIST(self.data_dir, train=False, download=True)
        FashionMNIST(self.data_dir, train=True, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            fmnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform_train)
            self.fmnist_train, self.fmnist_val = random_split(fmnist_full, [55000, 5000])
        if stage == "test":
            self.fmnist_test = FashionMNIST(self.data_dir, train=False, transform=self.transform_test)
        if stage == "predict":
            self.fmnist_predict = FashionMNIST(self.data_dir, train=False, transform=self.transform_test)
        

    def train_dataloader(self):
        return DataLoader(self.fmnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.fmnist_val, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.fmnist_test, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.fmnist_predict, batch_size=self.batch_size)
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(784, 392)),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(0.25)),
            ('fc12', nn.Linear(392, 196)),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(0.25)),
            ('fc3', nn.Linear(196, 98)),
            ('relu3', nn.ReLU()),
            ('drop3', nn.Dropout(0.25)),                                       
            ('fc4', nn.Linear(98, 49)),
            ('relu4', nn.ReLU()),
            ('output', nn.Linear(49, 10)),
            ('logsoftmax', nn.LogSoftmax(dim=1))
        ]))

        for m in self.model:
            if isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                variance = np.sqrt(2.0/(fan_in + fan_out))
                m.weight.data.normal_(0.0, variance)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(OrderedDict([
            ('bn', nn.BatchNorm2d(1)),
            ('conv1', nn.Conv2d(1, 64, 5, 1, 2)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout(0.1)),
            ('conv2', nn.Conv2d(64, 64, 5, 1, 2)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2)),
            ('dropout2', nn.Dropout(0.3)),
        ]))
        self.linear_layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 7 * 7, 256)),
            ('relu3', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(256, 64)),
            ('relu4', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(64)),
            ('fc3', nn.Linear(64, 10)),
            ('logsoftmax', nn.LogSoftmax(1)),
        ]))

        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.linear_layers:
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                variance = np.sqrt(2.0/(fan_in + fan_out))
                m.weight.data.normal_(0.0, variance)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    

class LitModel(pl.LightningModule):
    def __init__(self, model, optimizer, lr, checkpoint):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        self.log("val_loss", loss)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics
    
    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torch.sum(torch.argmax(y_hat, 1) == y) / len(y)
        return loss, acc
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer
    

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    

@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig):
    # To access elements of the config
    optimizer = cfg.optimizer
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.lr
    data_dir = cfg.data_dir
    architecture = cfg.architecture
    checkpoint = os.path.dirname(cfg.checkpoint)
    logs = cfg.logs
    
    if optimizer != 'adam' and optimizer != 'sgd':
        optimizer = 'adam'

    if architecture == 'linear':
        model = LitModel(Net(), optimizer, lr, checkpoint)
    elif architecture == 'conv':
        model = LitModel(CNN(), optimizer, lr, checkpoint)
    else:
        architecture = 'conv'
        model = LitModel(CNN(), optimizer, lr, checkpoint)
    dm = FMNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()
    dm.setup("fit")

    logger = TensorBoardLogger(save_dir=logs)
    trainer = pl.Trainer(accelerator="auto", max_epochs=epochs, default_root_dir=checkpoint, logger=logger)
    trainer.fit(model=model, datamodule=dm)

    torch.save(trainer.model.state_dict(), os.path.join(checkpoint, architecture + "_" + optimizer + "_lr" + str(lr) + "_" + str(epochs) + "e.pth"))


if __name__ == '__main__':
    main()
