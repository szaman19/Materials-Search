import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import numpy as np
import dataset as MOFdata

import random
from basic_net import BasicModel

# Everything starts in the main() function

def filter_data(x):
    # Clip anything below a and above b, map the range to -1 to 1
    a, b = -4000, 5000
    mid = (a + b)/2
    dif = b-a
    energy = np.clip(x[0] - mid, a - mid, b - mid) / (dif / 2)
    return np.array([energy, x[1], x[2]])


def filter_labels(x):
    return x[:3]

def load_datasets(
    grid_file = "data/probability.npy",
    csv_file = "data/ASR.csv",
    lattice_file = "data/grids.lattice.npy",
    feature = "lattice",
    seed=42
):
    # create training, validation, and test dataloaders of 90%, 5%, and 5% of data
    dataset = MOFdata.Dataset(grid_file, csv_file, lattice_file, feature, transform=filter_data, transform_labels=filter_labels)
    train_set_size = int(.9 * len(dataset))
    validation_set_size = int(.05 * len(dataset))
    test_set_size = len(dataset) - train_set_size - validation_set_size
    
    train_set, validation_set, test_set = random_split(
        dataset=dataset,
        lengths=(train_set_size,
        validation_set_size,
        test_set_size,
        ),
        generator=torch.Generator().manual_seed(seed))

    loader_args = dict(batch_size=64, num_workers=4)
    train_loader = DataLoader(train_set, **loader_args)
    validation_loader = DataLoader(validation_set, **loader_args)
    test_loader = DataLoader(test_set, **loader_args)
    return train_loader, validation_loader, test_loader


def loss_fn(output, target):
    ### Just MSE loss, can be changed in every place it is used easily here
    return nn.functional.mse_loss(output, target)

def proportional_loss(output, target):
    ### A loss function for determining relative distance of output to target
    return torch.mean(torch.abs(output-target)/torch.abs(target))

class ModulePL(pl.LightningModule):
    ### the LightningModule for training
    def __init__(self, features=3, channels=3, dropout=0.2, device=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = BasicModel(features=features, channels=channels, dropout=dropout)
        self.activations = {}
        for name, child in self.model.named_children():
            child.register_forward_hook(self.get_activation(name))
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # forward step
        y_hat = self.forward(x)
        # loss (backprop done by pytorch-lightning)
        loss = loss_fn(y.float(), y_hat)
        self.log("train_loss", loss)
        return loss
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
        

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.float()
        y_hat = self.forward(x)
        loss = loss_fn(y_hat, y)
        l1_loss = nn.functional.l1_loss(y_hat, y)

        # log various types of loss
        self.log('validation_loss', loss)
        self.log('validation_l1_loss', l1_loss)
        self.log('validation_p_loss', proportional_loss(y_hat, y))
        for name, values in self.activations.items():
            # log activations
            self.log(f'activations/{name}-max', torch.max(torch.abs(values)).item())
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3, amsgrad=True)
        optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.99, nesterov=True)
        # optimizer = optim.Adadelta(self.parameters(), lr=1e-3)
        return dict(optimizer=optimizer, lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, cooldown=20, verbose=True), monitor="validation_loss")
        # return optimizer


def main():
    model = ModulePL()

    # set up trainer
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    
    # get dataloaders for each dataset
    train_loader, validation_loader, test_loader = load_datasets()

    # set up logging and checkpoints
    wandb_logger = WandbLogger(project="lattice", log_model="all")
    chkpt_dir = "./lattice_pt"
    checkpoint_callback = ModelCheckpoint(dirpath=chkpt_dir, save_top_k=2, monitor="validation_loss")

    wandb_logger.watch(model.model)
    # train
    trainer = pl.Trainer(
        logger = wandb_logger,
        limit_train_batches=100,
        limit_val_batches=10,
        max_epochs=100,
        gradient_clip_val=0.5,
        track_grad_norm=2,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
    )

    # train the model
    trainer.fit(model=model, train_dataloaders=train_loader,  val_dataloaders=validation_loader)

if __name__ == "__main__":
    main()